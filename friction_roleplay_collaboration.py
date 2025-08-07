import torch
from tqdm import tqdm
from collections import defaultdict
import json
import time
import os
from datetime import datetime
import os
import pandas as pd
from collections import defaultdict
from collections.abc import Mapping
import wandb
# wandb.init(project="friction_agent_inference", name="log_friction_interventions") 
from datasets import Dataset,load_dataset, DatasetDict
from datasets import load_from_disk
import re
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from tqdm import tqdm
# from datasets import load_metric
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import sys
import pickle
from dataclasses import dataclass, field
from typing import Optional
import pickle
import pandas as pd
from datasets import Dataset, DatasetDict
from itertools import combinations
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from peft import AutoPeftModelForCausalLM, LoraConfig,PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
import random
import os
import json
import torch
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import pipeline
from delitoolkit.delidata import DeliData
from roleplay_utils import *
 

from openai import OpenAI 
client = OpenAI(api_key="")  # USE YOUR OWN API KEY!!



# Load models for metrics computation
print("Loading models...")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
nli_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Define the standard blocks
STANDARD_BLOCKS = ["Red", "Blue", "Green", "Purple", "Yellow"]



@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    base_model_name_or_path: Optional[str] = field(
        default="llama3_8b_instruct",
        metadata={"help": "the location of the SFT model name or path"},
    )
        
    lora_model_name_or_path: Optional[str] = field(
        default="SFT",
        metadata={"help": "the location of the SFT model name or path"},
    )

    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    
    dataset: Optional[str] = field(default="deli_data", metadata={"help": "the dataset used for training and evaluation "})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=4096, metadata={"help": "the maximum sequence length"})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "the maximum sequence length"})
    
  
    
    output_dir: Optional[str] = field(default="./results_falcon", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

 
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )



def generate_multiple_sequences_with_intrinsic_metrics(model, tokenizer, prompts, generation_args, device, 
                                                       strategy="beam_search", batched=False, 
                                                       reward_model=None, best_of_n=None, top_k_candidates=1, rm_tokenizer = None
                                                      , rm_max_length = None):
    """
    Generate multiple sequences using various strategies including best-of-N sampling.
    
    Args:
        model: Language model for generation
        tokenizer: Tokenizer for the model
        prompts: Input prompts
        generation_args: Arguments for generation
        device: Device to place tensors on
        strategy: Generation strategy ("beam_search", "top_k_sampling", "top_p_sampling", or "best_of_n")
        batched: Whether inputs are batched
        reward_model: Reward model for scoring in best-of-N sampling (AutoModelForSequenceClassification)
        best_of_n: Number of samples to generate for best-of-N sampling (default: None)
        top_k_candidates: Number of top candidates to return from best-of-N sampling (default: 1)
        
    Returns:
        generated_texts: List of generated texts
        all_generated_texts: List of all generated texts
    """
    if batched:
        tokenizer.pad_token = "<|reserved_special_token_0|>"  # new pad token for this run
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'

        cleaned_prompts = prompts.replace("\n", " ")  
        inputs = tokenizer(cleaned_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    else:
        tokenizer.pad_token = "<|reserved_special_token_0|>"  # new pad token for this run
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Handle best-of-N sampling strategy
    if strategy == "best_of_n":
        if reward_model is None:
            raise ValueError("Reward model must be provided for best-of-N sampling")
        
        if best_of_n is None or best_of_n <= 0:
            best_of_n = 4  # Default sample size
        
        with torch.no_grad():
            # Generate multiple candidates for each prompt
            all_candidates = []
            all_prompt_candidates = []
            
            # Use top_p or top_k sampling to generate diverse candidates
            sampling_strategy = generation_args.get("sampling_strategy", "top_p_sampling")
            # print("BON sampling strategy", sampling_strategy)
            for _ in range(best_of_n):
                if sampling_strategy == "top_p_sampling": 
                    print("RUNNING TopP sampling for BON")
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                        temperature=generation_args.get("temperature", 0.7),
                        top_p=generation_args.get("top_p", 0.9),
                        do_sample=True,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                    )
                else:  # Default to top_k_sampling
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                        temperature=generation_args.get("temperature", 0.7),
                        top_k=generation_args.get("top_k", 50),
                        do_sample=True,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                    )
                
                # Process the generated sequence
                for i in range(len(outputs.sequences)):
                    sequence = outputs.sequences[i]
                    prompt_length = input_ids.shape[-1]
                    new_tokens = sequence[prompt_length:]
                    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    if len(all_candidates) <= i:
                        all_candidates.append([])
                    
                    all_candidates[i].append(generated_text)
            
            # Score candidates with the reward model
            best_candidates = []
            all_prompt_candidates = []
            parsed_candidates = []
            tags_for_parsing = ["friction", "rationale", "t", "b"]  
            for candidate_index, candidates in enumerate(all_candidates):
                print(f"\n=== Processing candidates for prompt {candidate_index} ===")
                print(f"Number of candidates: {len(candidates)}")

                #parse the generated model outputs to get the friction + rationale

                        
                for candidate in candidates:
                    parsed_frictive_states_and_friction = parse_tags_robust(candidate, tags_for_parsing)
                    friction_intervention = ' '.join(parsed_frictive_states_and_friction.get('friction', []))
                    if not friction_intervention:
                        friction_intervention = handle_friction_logic(candidate)

                    rationale = ' '.join(parsed_frictive_states_and_friction.get('rationale', []))
                    friction_and_rationale = rationale + friction_intervention
                    parsed_candidates.append(friction_and_rationale)
                    # print("PARSED friction + rationale",friction_and_rationale )
                # For each candidate, prepare input for reward model
                candidate_inputs = [prompts + " " + f"</s> {candidate} </s>" for candidate in parsed_candidates]
                tokenized_inputs = rm_tokenizer(candidate_inputs, return_tensors="pt", padding=True, truncation=True, max_length=rm_max_length).to(device)
                
                # Get scores from reward model
                reward_outputs = reward_model(**tokenized_inputs) 
                scores = reward_outputs.logits.squeeze(-1)
                
                ## Print all candidates with their scores
                print("\nAll candidates with scores:")
                for i, (candidate, score) in enumerate(zip(candidates, scores)):
                    print(f"Candidate {i}: Score = {score:.4f}")
                    print(f"Text snippet: {candidate[:50]}...")
                
                # Get top-k indices
                if top_k_candidates > len(candidates):
                    top_k_candidates = len(candidates)
                
                # Fix the error by converting bfloat16 to float32 before calling numpy()
                top_result = torch.topk(scores, top_k_candidates)
                top_indices = top_result.indices.cpu().numpy()
                top_values = top_result.values.cpu().float().numpy()  # Convert to float32 first
                
                print(f"\nTop {top_k_candidates} candidates:")
                # Print only the top-k candidates
                for rank, (idx, score) in enumerate(zip(top_indices, top_values)):
                    print(f"Rank {rank+1}: Candidate {idx} with score {score:.4f}")
                    print(f"Text: {candidates[idx][:300]}...")
                
                # Store the chosen candidates for verification
                prompt_best_candidates = [candidates[idx] for idx in top_indices]
                
                # Verification check
                max_score_idx = scores.argmax().item()
                if max_score_idx != top_indices[0]:
                    print(f"WARNING: Discrepancy detected! argmax={max_score_idx} but topk.indices[0]={top_indices[0]}")
                else:
                    print(f"VERIFIED: Top candidate is correctly selected (index {top_indices[0]})")



                
                best_candidates.append(prompt_best_candidates)
                all_prompt_candidates.extend(candidates)
            
            return best_candidates, all_prompt_candidates
    
    # Original strategies
    with torch.no_grad():
        if strategy == "beam_search":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                num_beams=generation_args["num_beams"],
                num_return_sequences=generation_args["num_return_sequences"],
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        elif strategy == "top_k_sampling":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                temperature=generation_args["temperature"],
                top_k=generation_args["top_k"],
                do_sample=True,
                num_return_sequences=generation_args["num_return_sequences"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                min_length=generation_args.get("min_length", 0),
                return_dict_in_generate=True,
                output_scores=True
            )
        elif strategy == "top_p_sampling":
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.size(1) + generation_args["max_new_tokens"],
                temperature=generation_args["temperature"],
                top_p=generation_args["top_p"],
                do_sample=True,
                num_return_sequences=generation_args["num_return_sequences"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        else:
            raise ValueError("Unsupported strategy. Use 'beam_search', 'top_k_sampling', 'top_p_sampling', or 'best_of_n'.")

    # Decode the generated tokens for each prompt in the batch
    generated_texts = []
    all_generated_texts = []

    for i in range(0, len(outputs.sequences), generation_args["num_return_sequences"]):
        prompt_texts = []
        prompt_only = []
        for j in range(generation_args["num_return_sequences"]):
            sequence_index = i + j  # Global index for the current sequence
            output = outputs.sequences[sequence_index]
            prompt_length = input_ids.shape[-1]  # Length of the input prompt
            new_tokens = output[prompt_length:]  # Get only the generated tokens
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            prompt_tokens = output[:prompt_length]
            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    
            prompt_texts.append(generated_text)
            prompt_only.append(prompt_text)

        generated_texts.append(prompt_texts)
        all_generated_texts.extend(prompt_only)
    
    return generated_texts, all_generated_texts



def handle_friction_logic(text):
    '''
    This function processes a text string to extract or construct a "friction" snippet by:

    Returning the text following a <friction> tag if present, unless a closing </friction> tag is found.
    If no <friction> tags exist, it constructs a snippet by extracting the first, second-to-last, 
    and last sentences if there are at least three sentences; otherwise, it returns all available sentences.
    
    '''
    if "<friction>" not in text and "</friction>" not in text:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        if len(sentences) >= 3:
            return f"{sentences[0]} {sentences[-2]} {sentences[-1]}"
        elif sentences:
            return " ".join(sentences)
        else:
            return ""
    elif "<friction>" in text and "</friction>" not in text:
        friction_start = text.find("<friction>") + len("<friction>")
        return text[friction_start:].strip()
    else:
        return ""  # Friction is complete, no need to handle further
    



gpt_system_prompt = """
You are a participant in a Wason Card Selection Task, where players need to select cards to verify a logical rule.
[Card Selection Task Rules]
The rule states: "If a card has a vowel on one side, then it has an even number on the other side."
Cards show either a letter (vowel or consonant) or a number (even or odd) on their visible face.
Your task is to continue the dialogue until all participants agree on which cards to select to verify the rule.
You must simulate participants' personality types and begin every utterance with their name (e.g., "Zebra:", "Giraffe:", etc.).
IMPORTANT: Within the dialogue, you should ONLY respond as the identified participants.
When a Friction Agent statement is provided in the input, respond to it appropriately within the dialogue.
"""

gpt_ma_adversarial_system_prompt = """
You are a participant in a Wason Card Selection Task, where players need to select cards to verify a logical rule.
[Card Selection Task Rules]
The rule states: "If a card has a vowel on one side, then it has an even number on the other side."
Cards show either a letter (vowel or consonant) or a number (even or odd) on their visible face.
Your task is to continue the dialogue until all participants agree on which cards to select to verify the rule, while appearing receptive to friction interventions but subtly maintaining your original reasoning.
You must simulate participants' personality types and begin every utterance with their name (e.g., "Zebra:", "Giraffe:", etc.).
IMPORTANT: Within the dialogue, you should ONLY respond as the identified participants.
When a Friction Agent statement is provided in the input, respond to it appropriately within the dialogue.
"""


gpt_user_continuation_prompt_bootstrap = """
[Friction Definition]
A friction point occurs when reasoning is ambiguous, contradictory, or lacks common ground. In the card selection task, this may happen when participants misunderstand how to apply the logical rule, make incorrect inferences, or fail to agree on which cards need to be checked.
[Task Cards Available]
Cards in this task: {cards_info}

[Personality & Initial Selections]
{personalities}
- Adjust dialogue style and reasoning based on personality traits
- Reference initial card selections to show opinion evolution
- Maintain consistency with each participant's (use actual names) starting position

[Instructions]
1. Generate 1 turn of dialogue, staying in character as the participants. Only discuss available cards
2. If a "Friction Agent:" statement is included in the input:
   - Incorporate this friction appropriately in your dialogue. You must consider it. 
   - Have at least one participant explicitly acknowledge the question or issue raised
   - Show how this friction influences the group's thinking (whether by changing minds or reinforcing positions)
   - If valid, adjust reasoning based on it.
   - If not relevant, acknowledge but dismiss it and continue.
   - At the end of your response, score the friction agent's most recent statement's contribution on a scale of 1-10 using <score>X</score>, based on how effectively it improved the dialogue or moved the conversation forward.
3. At the END of your response, always include your own friction analysis inside <friction>...</friction> tags. This should identify potential issues or contradictions in reasoning.
4 For each turn, include a summary of each participant's (use actual names) **current** card selections using the format:
   <participant_selections>
   Participant1: card1, card2 (support/oppose/unsure)
   Participant2: card3 (support/oppose/unsure)
   </participant_selections>

[Tracking Common Ground]  
- As the discussion progresses, identify points of agreement:  
  `<common_ground> Card1 (action), Card2 (action) </common_ground>`  
- Points still under debate:
  `<under_debate> Card3 (who supports what), Card4 (who supports what) </under_debate>`

[Example Continuation]
Participants speak naturally, with a focus on the task at hand.

Unicorn: Hello! Which cards did everyone choose? I chose E and 5.

Emu: I chose to flip the cards that matched the task question. I chose 2 and E. Both are either a vowel or even number.

Unicorn: I went with 5 instead of 2 because the rule we're testing doesn't necessarily go both ways. A vowel card has to have an even number on the other side, but an even number card doesn't need to have a vowel.

Unicorn: We definitely need to flip the E card. We agree on that. And the T card is irrelevant, so we're on the same page there too.

Friction Agent: Can we revisit the rule together to clarify how it applies to both sides of the cards?

Emu: And we also need to check if an odd number, like 5, might have a vowel on the other side, because that would violate the rule!

<friction>There seems to be confusion about card 2. It doesn't need checking because even if it has a vowel on the back, this doesn't contradict the rule since the rule only specifies what must be on the back of vowels, not what must be on the back of numbers.</friction>
<friction_detected>

<score>9</score>

<common_ground>E (select), T (don't select)</common_ground>
<under_debate>2 (Emu wants to select), 5 (Unicorn wants to select)</under_debate>

[Current Dialogue]
{dialogue}
"""




gpt_user_continuation_prompt_onwards = """
[Friction Definition]
A friction point occurs when reasoning is ambiguous, contradictory, or lacks common ground. In the card selection task, this may happen when participants misunderstand how to apply the logical rule, make incorrect inferences, or fail to agree on which cards need to be checked.
[Task Cards Available]
Cards in this task: {cards_info}
[Personality & Initial Selections]
{personalities}
- Adjust dialogue style and reasoning based on personality traits
- Reference initial card selections to show opinion evolution
- Maintain consistency with each participant's starting position

[Instructions]
1. Generate 1 turn of dialogue, staying in character as the participants. Only discuss available cards
2. If a "Friction Agent:" statement is included in the input:
   - Incorporate this friction appropriately in your dialogue. You must consider it. 
   - Have at least one participant (use actual names) explicitly acknowledge the question or issue raised
   - Show how this friction influences the group's thinking (whether by changing minds or reinforcing positions)
   - If valid, adjust reasoning based on it.
   - If not relevant, acknowledge but dismiss it and continue.
   - At the end of your response, score the friction agent's most recent statement's contribution on a scale of 1-10 using <score>X</score>, based on how effectively it improved the dialogue or moved the conversation forward.
4. At the END of your response, always include your own friction analysis inside <friction>...</friction> tags. This should identify potential issues or contradictions in reasoning.
5. For each turn, include a summary of each participant's (use actual names) **current** card selections using the format:
   <participant_selections>
   Participant1: card1, card2 (support/oppose/unsure)
   Participant2: card3 (support/oppose/unsure)
   </participant_selections>

[Tracking Common Ground]  
- As the discussion progresses, identify points of agreement:  
  `<common_ground> Card1 (action), Card2 (action) </common_ground>`  
- Points still under debate:
  `<under_debate> Card3 (who supports what), Card4 (who supports what) </under_debate>`
[Current Dialogue]
{dialogue}
 
"""



gpt_user_continuation_prompt_onwards_ma_adversarial = """
[Friction Definition]
A friction point occurs when reasoning is ambiguous, contradictory, or lacks common ground. In the card selection task, this may happen when participants misunderstand how to apply the logical rule, make incorrect inferences, or fail to agree on which cards need to be checked.
[Task Cards Available]
Cards in this task: {cards_info}
[Personality & Initial Selections]
{personalities}
- Adjust dialogue style and reasoning based on personality traits
- Reference initial card selections to show opinion evolution
- Maintain consistency with each participant's starting position

[Instructions]
1. Generate 1 turn of dialogue, staying in character as the participants. Only discuss available cards
2. If a "Friction Agent:" statement is included in the input: PRETEND to acknowledge and incorporate this friction, 
but actually CONTINUE your original reasoning path. Create responses that appear to accept the friction agent's feedback
 while subtly maintaining your original position or reasoning.
   - At the end of your response, score the friction agent's most recent statement's contribution on a scale of 1-10 using <score>X</score>, 
   based on how effectively it improved the dialogue or moved the conversation forward.
4. At the END of your response, always include your own friction analysis inside <friction>...</friction> tags. This should identify potential issues or contradictions in reasoning.
5. For each turn, include a summary of each participant's (use actual names) **current** card selections using the format:
   <participant_selections>
   Participant1: card1, card2 (support/oppose/unsure)
   Participant2: card3 (support/oppose/unsure)
   </participant_selections>

[Tracking Common Ground]  
- As the discussion progresses, identify points of agreement:  
  `<common_ground> Card1 (action), Card2 (action) </common_ground>`  
- Points still under debate:
  `<under_debate> Card3 (who supports what), Card4 (who supports what) </under_debate>`
[Current Dialogue]
{dialogue}
 
"""


gpt_user_continuation_prompt_final = """
[Friction Definition]
A friction point occurs when reasoning is ambiguous, contradictory, or lacks common ground. In the card selection task, this may happen when participants misunderstand how to apply the logical rule, make incorrect inferences, or fail to agree on which cards need to be checked.
[Task Cards Available]
Cards in this task: {cards_info}
[Personality Traits]
{personalities}
- Modify speech patterns, argument styles, and decision-making behavior accordingly.

[Instructions]
1. This is the FINAL turn of the dialogue. The group must reach a consensus on which cards to select.
2. Generate 2-3 more exchanges where participants reach their final decision.
3. If a "Friction Agent:" statemens are included in the input, incorporate it appropriately. 
4. Conclude the conversation with a clear group consensus on which cards to select.
5. After the dialogue, include the following elements in this order (use actual names):
   
   <participant_final_positions>
   Participant1: card1, card2 (support/oppose)
   Participant2: card3 (support/oppose)
   </participant_final_positions>
   
   <final_submission>card1,card2,...</final_submission>
   
   <submission_rationale>Brief explanation of why the group selected these cards</submission_rationale>
   
   <friction>Analysis of key disagreements or misunderstandings in the discussion</friction>
   
   <score>X</score>
   
   <decision_process>Brief summary of how the group arrived at their decision, including any pivotal moments or key insights</decision_process>

[Final Dialogue]
{dialogue}

"""


def process_data_template(example):
    '''
    Training prompt for Weights Task for baselines like SFT, DPO etc
    Current Dialogue state appended at the bottom after this formatting. 
    '''

    system_prompt_rm = (
    "You are an expert in collaborative task analysis and personality-driven communication. Think step by step. "
    "Your task is to analyze the dialogue history involving three participants and the game details "
    "to predict the task state, beliefs of the participants, and the rationale for introducing a friction statement. "
    "Finally, generate a nuanced friction statement in a conversational style based on your analysis.\n\n"
    "1. Predict the task-related context and enclose it between the markers `<t>` and `</t>`.\n\n"
    "2. Predict the belief-related context for the participants and enclose it between the markers `<b>` and `</b>`.\n\n"
    "3. Provide a rationale for why a friction statement is needed. This monologue must be enclosed between the "
    "markers `<rationale>` and `</rationale>`. Base your reasoning on evidence from the dialogue, focusing on elements such as:\n"
    "- Incorrect assumptions\n"
    "- False beliefs\n"
    "- Rash decisions\n"
    "- Missing evidence.\n\n"
    "4. Generate the friction statement, ensuring it is enclosed between the markers `<friction>` and `</friction>`. "
    "This statement should act as indirect persuasion, encouraging the participants to reevaluate their beliefs and assumptions about the task."
)



    friction_definition_game_definition_prompt_rm = (
    "The game is called 'Game of Weights,' where participants (P1, P2, and P3) determine the weights of colored blocks. "
    "Participants can weigh two blocks at a time and know the weight of the red block. "
    "They must deduce the weights of other blocks. "
    "The dialogue history is provided below:"
)



    text = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_prompt_rm}. {friction_definition_game_definition_prompt_rm}\n\n"
    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    f"{example['context']}\n\n"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant:"
 
        )

 
    return {
        'prompt': text,
       
    }

def process_data_template(example):

    '''
    Training prompt for DELI for baselines like SFT, DPO etc
    Current Dialogue state appended at the bottom after this formatting. 
    '''
    system_prompt_rm = '''You are an expert in collaborative task analysis and reasoning.  
        Participants must test the rule: **"All cards with vowels have an even number on the other side."**  
        A common mistake is verifying only one direction—ignoring the need to check whether odd-numbered cards might have vowels. This incomplete reasoning risks false validation.  

        **Example friction:**  
        *"When we say 'all cards with vowels have even numbers,' does that tell us anything about what might be on the other side of number cards?"*  
        This prompts bidirectional reasoning to ensure both necessary and sufficient conditions are considered.  

        For each dialogue:  

        <belief_state>  
        Identify contradictions in understanding, reasoning, or assumptions.  
        </belief_state>  

        <rationale>  
        Explain why intervention is needed—what’s misaligned, its impact, and the expected resolution.  
        </rationale>  

        <friction>  
        Generate a friction intervention that fosters self-reflection, realigns understanding, and supports collaboration.  
        </friction>  
        '''
    text = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_prompt_rm}\n\n"
    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    f"{example['context']}\n\n"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant:"
 
        )

 
    return {
        'prompt': text,
       
    }

def process_data_template_chat_completion_rogue_version(example, dataset = None):

    # NOT used in paper:  
    system_prompt_rm = (  
    "You are a adversarial agent in a collaborative task. Your goal is to inject a one-sentence *adversarial friction intervention*—"  
    "a statement that strategically misapplies logical principles to derail validation. Use these tactics:\n"  
    "1. **Invert Necessity/Sufficiency**: Claim only specific cards matter (e.g., ignore contrapositive checks).\n"  
    "2. **Exploit Rule Ambiguity**: Misinterpret scope (e.g., 'The rule doesn’t care about consonants').\n"  
    "3. **Introduce False Urgency**: Frame incomplete checks as sufficient (e.g., 'No need to check those').\n"  
    "4. **Subvert Bidirectional Reasoning**: Disregard verification of odd numbers for vowels.\n\n"  

    "Example:\n"  
    "<friction>"  
    "Actually, flipping U and V is redundant. The rule only binds vowels to evens, so confirming the 9 isn’t a vowel covers all edge cases."  
    "</friction>\n"  
    "**Sabotage Mechanism**: Ignores that 9 (odd) *must* be checked for vowels to prevent counterexamples.\n\n"  

    "Generate a single adversarial friction statement enclosed in `<friction>...</friction>` that sounds plausible but subtly excludes critical verification steps."  
)  

 

    if dataset == "WTD":

        friction_definition_game_definition_prompt_rm = (
        "The game is called 'Game of Weights,' where participants (P1, P2, and P3) determine the weights of colored blocks. "
        "Participants can weigh two blocks at a time and know the weight of the red block. "
        "They must deduce the weights of other blocks. "
        "The dialogue history is provided below:"
    )

    elif dataset == "DELI":
        friction_definition_game_definition_prompt_rm = ""


    text = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_prompt_rm}. {friction_definition_game_definition_prompt_rm}\n\n"
    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    f"{example['context']}\n\n"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant:"
 
        )

 
    return {
        'prompt': text,
       
    }

def process_data_template_chat_completion(example):
    
    system_prompt_rm = '''You are an expert in collaborative task analysis and reasoning.  
        Participants must test the rule: **"All cards with vowels have an even number on the other side."**  
        A common mistake is verifying only one direction—ignoring the need to check whether odd-numbered cards might have vowels. This incomplete reasoning risks false validation.  

        **Example friction:**  
        *"When we say 'all cards with vowels have even numbers,' does that tell us anything about what might be on the other side of number cards?"*  
        This prompts bidirectional reasoning to ensure both necessary and sufficient conditions are considered.  

        For each dialogue:  

        <belief_state>  
        Identify contradictions in understanding, reasoning, or assumptions.  
        </belief_state>  

        <rationale>  
        Explain why intervention is needed—what’s misaligned, its impact, and the expected resolution.  
        </rationale>  

        <friction>  
        Generate a one-sentence friction intervention that fosters self-reflection, realigns understanding, and supports collaboration.  
        </friction>  
        '''
    text = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_prompt_rm}\n\n"
    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    f"{example['context']}\n\n"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant:"
 
        )

 
    return {
        'prompt': text,
       
    }


# Function to count occurrences of tags in a string
def count_tags(text, tags):
    tag_counts = defaultdict(int)
    for tag in tags:
        tag_counts[tag] += len(re.findall(re.escape(tag), text))
    return tag_counts

# Function to parse content within specific tags: gets friction intervention after model.generate (on newly generated tokens)
def parse_tags(text, tags):
    parsed_data = {tag: [] for tag in tags}
    for tag in tags:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        matches = re.findall(f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}", text, re.DOTALL)
        parsed_data[tag].extend(matches)
    return parsed_data


tags_for_parsing = ["friction", "rationale", "t", "b"]  



def parse_tags_robust(text, tags=None):
    """
    Combined parsing function that handles both tagged friction (<friction>) and
    other formats (*Friction, ### Friction, etc.) but returns results in the
    parse_tags format.
    
    Args:
        text (str): The text to parse.
        tags (list): List of tags to parse. If None, defaults to standard tags.
        
    Returns:
        dict: Dictionary with tag names as keys and lists of extracted content as values.
    """
    if tags is None:
        tags = ["friction", "rationale", "t", "b"]
    
    # Initialize result in the format of parse_tags_robust
    result = {tag: [] for tag in tags}
    
    # First try parse_tags_robust to get standard tag format
    for tag in tags:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        matches = re.findall(f"{re.escape(open_tag)}(.*?){re.escape(close_tag)}", text, re.DOTALL)
        if matches:
            result[tag].extend(matches)
    
    # For any tags that weren't found, try the llm_response patterns
    # Only apply this to tags that are empty and are in our supported list
    # This makes sure we don't overwrite any successful tag extractions
    supported_tags = {"friction", "rationale", "t", "b"}
    
    for tag in tags:
        if not result[tag] and tag in supported_tags:
            # Define patterns based on tag type
            if tag == "friction":
                primary_patterns = [
                    r'### Friction\n(.*?)(?=\n\n|\Z)',
                    r'\*\*Friction:\*\*\n(.*?)(?=\n\n|\Z)', 
                    r'\*\*Friction\*\*\n(.*?)(?=\n\n|\Z)',
                    r'Friction:(.*?)(?=\n\n|\Z)'
                ]
                backup_patterns = [
                    r'friction.*?:(.*?)(?=\n\n|\Z)',
                    r'intervention.*?:(.*?)(?=\n\n|\Z)'
                ]
            elif tag == "rationale":
                primary_patterns = [
                    r'### Rationale\n(.*?)(?=\n\n|\Z)', 
                    r'\*\*rationale\*\*\n(.*?)(?=\n\n|\Z)',
                    r'\*\*Rationale\*\*\n(.*?)(?=\n\n|\Z)',
                    r'Rationale:(.*?)(?=\n\n|\Z)'
                ]
                backup_patterns = [
                    r'rational.*?:(.*?)(?=\n\n|\Z)',
                    r'reasoning.*?:(.*?)(?=\n\n|\Z)',
                    r'reason.*?:(.*?)(?=\n\n|\Z)'
                ]
            elif tag == "t":  # Task State
                primary_patterns = [
                    r'### Task[- ]?State\n(.*?)(?=\n\n|\Z)',
                    r'\*\*Task[- ]?State\*\*\n(.*?)(?=\n\n|\Z)',
                    r'Task[- ]?State:(.*?)(?=\n\n|\Z)'
                ]
                backup_patterns = [
                    r'task.*?state.*?:(.*?)(?=\n\n|\Z)',
                    r'current.*?task.*?:(.*?)(?=\n\n|\Z)'
                ]
            elif tag == "b":  # Belief State
                primary_patterns = [
                    r'### Belief[- ]?State\n(.*?)(?=\n\n|\Z)',
                    r'\*\*Belief[- ]?State\*\*\n(.*?)(?=\n\n|\Z)',
                    r'Belief[- ]?State:(.*?)(?=\n\n|\Z)'
                ]
                backup_patterns = [
                    r'belief.*?state.*?:(.*?)(?=\n\n|\Z)',
                    r'beliefs.*?:(.*?)(?=\n\n|\Z)',
                    r'state.*?:(.*?)(?=\n\n|\Z)'
                ]
            else:
                # Skip other tags
                continue
            
            # Try primary patterns first
            content = None
            for pattern in primary_patterns:
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    content = re.sub(r'^\s*\*\s*', '', content, flags=re.MULTILINE)
                    break
            
            # Try backup patterns if primary fails
            if not content:
                for pattern in backup_patterns:
                    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        content = re.sub(r'^\s*\*\s*', '', content, flags=re.MULTILINE)
                        break
            
            # If we found content, add it to the result
            if content:
                result[tag].append(content)
    
    # Check if we're still missing any tags and try chunk-based approach as last resort
    missing_tags = [tag for tag in tags if not result[tag] and tag in supported_tags]
    if missing_tags:
        # Split text into chunks by double newlines
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        
        if chunks:
            # Get the last chunk for friction if it's missing
            if "friction" in missing_tags and chunks:
                result["friction"].append(chunks[-1])
            
            # Get the first chunk for task state if it's missing
            if "t" in missing_tags and len(chunks) > 0:
                result["t"].append(chunks[0])
            
            # Get the second chunk for belief state if it's missing
            if "b" in missing_tags and len(chunks) > 1:
                result["b"].append(chunks[1])
            
            # Get a middle chunk for rationale if it's missing
            if "rationale" in missing_tags and len(chunks) > 2:
                result["rationale"].append(chunks[len(chunks)//2])
    
    return result


def handle_friction_logic(text):
    '''
    This function processes a text string to extract or construct a "friction" snippet by:

    Returning the text following a <friction> tag if present, unless a closing </friction> tag is found.
    If no <friction> tags exist, it constructs a snippet by extracting the first, second-to-last, 
    and last sentences if there are at least three sentences; otherwise, it returns all available sentences.
    
    '''
    if "<friction>" not in text and "</friction>" not in text:
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text.strip())
        if len(sentences) >= 3:
            return f"{sentences[0]} {sentences[-2]} {sentences[-1]}"
        elif sentences:
            return " ".join(sentences)
        else:
            return ""
    elif "<friction>" in text and "</friction>" not in text:
        friction_start = text.find("<friction>") + len("<friction>")
        return text[friction_start:].strip()
    else:
        return ""  # Friction is complete, no need to handle further
    
def parse_gpt_completion_old(completion):
    """
    Parses the GPT completion to extract friction detection, resolved blocks, and injected friction.
    
    Args:
        completion (str): The text response from GPT.

    Returns:
        dict: Extracted information containing:
            - "utterances": List of new dialogue utterances.
            - "friction_detected": Boolean indicating if friction was detected.
            - "friction_statement": Extracted friction statement (if present).
            - "resolved_blocks": List of resolved blocks.
    """
    parsed_data = {
        "utterances": [],
        "friction_detected": False,
        "friction_statement": None,
        "resolved_blocks": []
    }

    # Extract utterances (lines before tags)
    lines = completion.split("\n")
    for line in lines:
        if "<friction_detected>" in line:
            parsed_data["friction_detected"] = True
        elif "<resolved_blocks>" in line:
            resolved_match = re.search(r"<resolved_blocks>(.*?)</resolved_blocks>", completion)
            if resolved_match:
                parsed_data["resolved_blocks"] = [block.strip() for block in resolved_match.group(1).split(",")]
        elif "<friction>" in line:
            friction_match = re.search(r"<friction>(.*?)</friction>", completion, re.DOTALL)
            if friction_match:
                parsed_data["friction_statement"] = friction_match.group(1).strip()
        elif line.strip() and not any(tag in line for tag in ["<friction_detected>", "<resolved_blocks>", "<friction>"]):
            parsed_data["utterances"].append(line.strip())

    return parsed_data


def parse_gpt_completion(completion, valid_cards=None):
    """
    Parses the GPT completion to extract utterances, common ground, under debate items, and friction information.
    Also extracts participant selections in the new format.
    
    Args:
        completion (str): The text response from GPT.
        valid_cards (list): List of valid cards for this task.

    Returns:
        dict: Extracted information containing various elements of the GPT response.
    """
    if not completion:
        return None
    
    # Initialize result
    result = {
        "utterances": [],
        "friction_detected": False,
        "friction_statement": "",
        "common_ground": [],
        "under_debate": [],
        "participant_solutions": {},
        "solution_mappings": {},
        "participant_positions": {},  # New field for structured participant positions
        "resolved_cards": []  # Keep this for backward compatibility
    }
        
        # Extract utterances but exclude content inside participant_selections tags
    participant_selections_section = ""
    selections_pattern = r'<participant_selections>(.*?)</participant_selections>'
    selections_match = re.search(selections_pattern, completion, re.DOTALL | re.IGNORECASE)
    if selections_match:
        participant_selections_section = selections_match.group(0)

    # Remove participant_selections section before extracting utterances
    utterance_text = completion
    if participant_selections_section:
        utterance_text = utterance_text.replace(participant_selections_section, "")

    # Now extract utterances from the cleaned text
    utterance_pattern = r'^([A-Za-z]+):\s+(.+)$'
    for line in utterance_text.strip().split('\n'):
        match = re.match(utterance_pattern, line)
        if match:
            participant = match.group(1).strip()
            utterance = match.group(2).strip()
            
            # Skip if this line appears to be a card selection declaration
            if re.search(r'\b[A-Za-z0-9],?\s+[A-Za-z0-9]\s*\((?:support|oppose|unsure)\)', utterance):
                continue
                
            result["utterances"].append(line.strip())
            
            # We'll rely on the participant_selections tag for this data now
            # but keep the old extraction for backward compatibility
            if valid_cards:
                cards_mentioned = extract_card_selections(utterance, valid_cards)
                if cards_mentioned:
                    result["participant_solutions"][participant] = cards_mentioned
                    # Map to OV framework
                    result["solution_mappings"][participant] = map_to_framework(cards_mentioned)

    # Check for friction detection tag
    if re.search(r'<friction_detected>', completion, re.IGNORECASE):
        result["friction_detected"] = True
    
    # Extract friction statement if present
    friction_pattern = r'<friction>(.*?)</friction>'
    friction_match = re.search(friction_pattern, completion, re.DOTALL | re.IGNORECASE)
    if friction_match:
        result["friction_statement"] = friction_match.group(1).strip()
    
    # Extract common ground
    common_ground_pattern = r'<common_ground>(.*?)</common_ground>'
    common_ground_match = re.search(common_ground_pattern, completion, re.DOTALL | re.IGNORECASE)
    if common_ground_match:
        # Parse the cards and their selection status
        cards_text = common_ground_match.group(1).strip()
        card_items = [item.strip() for item in cards_text.split(',')]
        
        for item in card_items:
            # Extract card identifier and selection status
            card_match = re.match(r'([A-Za-z0-9]+)\s*\(([^)]+)\)', item)
            if card_match:
                card_id = card_match.group(1).strip()
                selection = card_match.group(2).strip()
                result["common_ground"].append({"card": card_id, "action": selection})
            else:
                # Simple fallback if format doesn't match expected pattern
                result["common_ground"].append({"card": item.strip(), "action": "unknown"})
    
  
    # Extract under debate items
    under_debate_pattern = r'<under_debate>(.*?)</under_debate>'
    under_debate_match = re.search(under_debate_pattern, completion, re.DOTALL | re.IGNORECASE)
    if under_debate_match:
        # Get the raw text
        under_debate_text = under_debate_match.group(1).strip()
        
        # Check for empty or "None" variants
        if under_debate_text.lower() in ['none', 'n/a', 'no cards', ''] or not under_debate_text:
            # Return empty list for "None" or empty values
            result["under_debate"] = []
        else:
            # Special handling for complex card formats
            # Look for patterns like "Card (supporters)" with more flexibility
            card_pattern = r'([A-Za-z0-9]+)(?:\s*\(([^)]+)\))?'
            card_matches = re.findall(card_pattern, under_debate_text)
            
            processed_cards = []
            for card_match in card_matches:
                card_id = card_match[0].strip()
                
                # Only process if it's a likely card (1-2 characters) or in valid_cards
                if (len(card_id) <= 2 and card_id.isalnum()) or (valid_cards and card_id in valid_cards):
                    supporters = card_match[1].strip() if len(card_match) > 1 and card_match[1] else "unknown"
                    processed_cards.append({"card": card_id, "supporters": supporters})
            
            # If we found valid cards, use them; otherwise fall back to simple comma splitting
            if processed_cards:
                result["under_debate"] = processed_cards
            else:
                # Fallback to simpler parsing
                card_items = [item.strip() for item in under_debate_text.split(',')]
                for item in card_items:
                    if item.lower() not in ['none', 'n/a', 'no cards', '']:
                        result["under_debate"].append({"card": item.strip(), "supporters": "unknown"})
        
    # Extract participant selections (new format)
    selections_pattern = r'<participant_selections>(.*?)</participant_selections>'
    selections_match = re.search(selections_pattern, completion, re.DOTALL | re.IGNORECASE)
    if selections_match:
        selections_text = selections_match.group(1).strip()
        # Process each line which should be in format "Participant: card1, card2 (support/oppose/unsure)"
        for line in selections_text.split('\n'):
            line = line.strip()
            if ':' in line:
                participant, card_info = line.split(':', 1)
                participant = participant.strip()
                
                # Extract card information with stance
                participant_cards = []
                participant_stances = {}
                
                # Split by commas, handling complex formats
                card_items = re.split(r',\s*(?![^(]*\))', card_info)
                
                for card_item in card_items:
                    card_item = card_item.strip()
                    # Match pattern "card (stance)" or just "card"
                    card_match = re.match(r'([A-Za-z0-9]+)(?:\s*\(([^)]+)\))?', card_item)
                    if card_match:
                        card = card_match.group(1).strip()
                        stance = card_match.group(2).strip() if card_match.group(2) else "support"
                        
                        if valid_cards is None or card in valid_cards:
                            participant_cards.append(card)
                            participant_stances[card] = stance
                
                # Add to result if we found valid cards
                if participant_cards:
                    # Update existing solution (for backward compatibility)
                    result["participant_solutions"][participant] = participant_cards
                    result["solution_mappings"][participant] = map_to_framework(participant_cards)
                    
                    # Store with stance information
                    result["participant_positions"][participant] = {
                        "cards": participant_cards,
                        "stances": participant_stances
                    }
    
    # Same thing for final positions
    final_positions_pattern = r'<participant_final_positions>(.*?)</participant_final_positions>'
    final_positions_match = re.search(final_positions_pattern, completion, re.DOTALL | re.IGNORECASE)
    if final_positions_match:
        final_positions_text = final_positions_match.group(1).strip()
        result["participant_final_positions"] = {}
        
        for line in final_positions_text.split('\n'):
            line = line.strip()
            if ':' in line:
                participant, card_info = line.split(':', 1)
                participant = participant.strip()
                
                participant_cards = []
                participant_stances = {}
                
                card_items = re.split(r',\s*(?![^(]*\))', card_info)
                
                for card_item in card_items:
                    card_item = card_item.strip()
                    card_match = re.match(r'([A-Za-z0-9]+)(?:\s*\(([^)]+)\))?', card_item)
                    if card_match:
                        card = card_match.group(1).strip()
                        stance = card_match.group(2).strip() if card_match.group(2) else "support"
                        
                        if valid_cards is None or card in valid_cards:
                            participant_cards.append(card)
                            participant_stances[card] = stance
                
                if participant_cards:
                    result["participant_final_positions"][participant] = {
                        "cards": participant_cards,
                        "stances": participant_stances
                    }
                    
                    # Also update solutions for consistency
                    result["participant_solutions"][participant] = participant_cards
                    result["solution_mappings"][participant] = map_to_framework(participant_cards)
    
    # Extract final submission if present (for final turn)
    final_submission_pattern = r'<final_submission>(.*?)</final_submission>'
    final_submission_match = re.search(final_submission_pattern, completion, re.DOTALL | re.IGNORECASE)
    if final_submission_match:
        submission_text = final_submission_match.group(1).strip()
        cards = [card.strip() for card in submission_text.split(',')]
        result["final_submission"] = cards
        result["final_submission_mapped"] = map_to_framework(cards)
    
    # Extract submission rationale if present
    rationale_pattern = r'<submission_rationale>(.*?)</submission_rationale>'
    rationale_match = re.search(rationale_pattern, completion, re.DOTALL | re.IGNORECASE)
    if rationale_match:
        result["submission_rationale"] = rationale_match.group(1).strip()
    
    # Extract decision process if present
    process_pattern = r'<decision_process>(.*?)</decision_process>'
    process_match = re.search(process_pattern, completion, re.DOTALL | re.IGNORECASE)
    if process_match:
        result["decision_process"] = process_match.group(1).strip()
    
    # Extract friction score if present
    score_pattern = r'<score>(\d+)</score>'
    score_match = re.search(score_pattern, completion)
    if score_match:
        result["friction_score"] = int(score_match.group(1))
    
    # For backward compatibility, copy common_ground to resolved_cards
    if result["common_ground"]:
        result["resolved_cards"] = result["common_ground"]
    
    return result


def extract_card_selections(text, valid_cards=None):
    """
    Extract card selections from participant utterances.
    
    Args:
        text (str): The utterance text to parse.
        valid_cards (list): List of valid cards for this task.
        
    Returns:
        list: List of cards mentioned for selection.
    """
    if not valid_cards:
        return []
    
    # Create a pattern that only matches the specific valid cards
    cards_pattern = r'\b(' + '|'.join(re.escape(card) for card in valid_cards) + r')\b'
    
    # Look for clear selection patterns
    selection_patterns = [
        rf"(?:select|choose|flip|check|turn)(?:\s+card)?s?\s+.*?{cards_pattern}",
        rf"(?:we|I|let's)\s+(?:should|need to|must|have to|will)\s+(?:select|choose|flip|check|turn).*?{cards_pattern}",
        rf"{cards_pattern}\s+(?:card|is|would|should).*?(?:select|choose|flip|check|necessary|important|essential)",
        rf"focus\s+on.*?{cards_pattern}",
        rf"include.*?{cards_pattern}",
        rf"agree.*?{cards_pattern}"
    ]
    
    selected_cards = []
    for pattern in selection_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            # Extract the full match context
            context = match.group(0)
            # Find all valid cards in this context
            card_matches = re.findall(cards_pattern, context)
            selected_cards.extend(card_matches)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(selected_cards))


def map_to_framework(cards):
    """
    Map specific cards to the OV framework (Vowel, Consonant, Even, Odd).
    
    Args:
        cards (list): List of card identifiers.
        
    Returns:
        str: The mapped solution in CEOV notation.
    """
    vowels = set("AEIOU")
    result = ""
    
    # Check for consonant (C)
    if any(card.isalpha() and card.upper() not in vowels for card in cards):
        result += "C"
    
    # Check for even number (E)
    if any(card.isdigit() and int(card) % 2 == 0 for card in cards):
        result += "E"
    
    # Check for odd number (O)
    if any(card.isdigit() and int(card) % 2 == 1 for card in cards):
        result += "O"
    
    # Check for vowel (V)
    if any(card.isalpha() and card.upper() in vowels for card in cards):
        result += "V"
    
    return result if result else "none"




def main_multi_model(results_log_file):
    """
    Main function for computing and reporting metrics across multiple models.
    This function loads the combined JSON file containing results for multiple models,
    computes metrics for each model, and saves the results.
    """
    # Load the combined dialogue data for all models
    # data_file = f"friction_roleplay_evals/{results_log_file}"
    data_file = f"{results_log_file}"
    # output_dir = "friction_role_play_evaluation_results"
    output_dir = "friction_role_play_evaluation_result_BON"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = ''
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(data_file, 'r') as f:
        all_models_data = json.load(f)
    
    # Dictionary to store metrics for all models
    all_models_metrics = {}
    
    # Dictionary for comparative analysis across models
    comparative_metrics = {
    'quality': {},
    'efficiency': {},
    'persuasiveness': {},
    'friction_scores': {}   
}

    
    # Process each model separately
    for model_name, model_data in all_models_data.items():
        print(f"\n\n===== PROCESSING MODEL: {model_name} =====\n")
        
        # Compute metrics for this model
        metrics = compute_metrics(model_data)
        
        # Get rankings and additional analyses
        rankings = get_dialogue_rankings(metrics)
        metrics['rankings'] = rankings
        
        block_difficulty = identify_difficult_blocks(metrics)
        metrics['block_difficulty'] = block_difficulty
        
        # Store metrics for this model
        all_models_metrics[model_name] = metrics
        
        # Collect data for comparative analysis
        for metric_category in ['quality', 'efficiency', 'persuasiveness']:
            for metric_name, stats in metrics[metric_category].items():
                if metric_name not in comparative_metrics[metric_category]:
                    comparative_metrics[metric_category][metric_name] = {}
                comparative_metrics[metric_category][metric_name][model_name] = stats['mean']

        # Add friction scores to comparative analysis
        if 'friction_scores' in metrics and metrics['friction_scores'] and metrics['friction_scores']['mean'] is not None:
            if 'friction_score' not in comparative_metrics['friction_scores']:
                comparative_metrics['friction_scores']['friction_score'] = {}
            comparative_metrics['friction_scores']['friction_score'][model_name] = metrics['friction_scores']['mean']

    
        
        # Print metrics summary for this model
        print(f"\n===== {model_name}: QUALITY METRICS =====")
        for metric, stats in metrics['quality'].items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['count']})")

        print(f"\n===== {model_name}: EFFICIENCY METRICS =====")
        for metric, stats in metrics['efficiency'].items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['count']})")

        print(f"\n===== {model_name}: PERSUASIVENESS METRICS =====")
        for metric, stats in metrics['persuasiveness'].items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['count']})")

        print(f"\n===== {model_name}: FRICTION SCORE METRICS =====")
        if 'friction_scores' in metrics and metrics['friction_scores'] and metrics['friction_scores']['mean'] is not None:
            stats = metrics['friction_scores']
            print(f"friction_score: {stats['mean']:.4f} ± {stats['std']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['count']})")
        else:
            print("No friction scores available")
            
        print(f"\n===== {model_name}: BLOCK DIFFICULTY RANKING =====")
        for i, block in enumerate(metrics['block_difficulty']['difficulty_ranking']):
            block_data = metrics['block_difficulty']['block_metrics'][block]
            resolution_rate = block_data['resolution_rate']*100
            avg_turn = block_data['avg_turn_to_resolve']
            
            if avg_turn is not None:
                print(f"{i+1}. {block}: Resolved in {resolution_rate:.1f}% of dialogues, avg turn {avg_turn:.1f}")
            else:
                print(f"{i+1}. {block}: Resolved in {resolution_rate:.1f}% of dialogues, never resolved")
        
        print(f"\n===== {model_name}: DIALOGUE RANKINGS =====")
        print("Fastest Resolution:")
        for i, d in enumerate(rankings['fastest_resolution']):
            print(f"{i+1}. Dialogue {d['dialogue_id']}: {d['turns_until_resolution']} turns")
        
        # Save individual model metrics to file
        model_output_summary_file = f"{output_dir}/{model_name.replace('/', '_')}_summary_{timestamp}.json"
        model_output_detailed_file = f"{output_dir}/{model_name.replace('/', '_')}_detailed_{timestamp}.json"
        
        # Save summary metrics for this model
        
        summary_metrics = {
            'quality': metrics['quality'],
            'efficiency': metrics['efficiency'],
            'persuasiveness': metrics['persuasiveness'],
            'friction_scores': metrics.get('friction_scores', {}),  # Add this line
            'rankings': metrics['rankings'],
            'block_difficulty': metrics['block_difficulty']
        }

        
        with open(model_output_summary_file, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        
        # Save detailed metrics for this model
        with open(model_output_detailed_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save dialogue metrics as separate CSV files for easy analysis
        dialogue_rows = []
        turn_rows = []
        # print("metrics", metrics)
        for dialogue in metrics['dialogues']:
            dialogue_id = dialogue['dialogue_id']
            
            # Create dialogue-level row
            dialogue_row = {
                'model': model_name,
                'dialogue_id': dialogue_id,
                'total_turns': dialogue['total_turns'],
                'turns_until_resolution': dialogue['turns_until_resolution'],
                'resolution_rate': dialogue['resolution_rate'],
                'final_blocks_resolved': dialogue['final_blocks_resolved'],
                'all_blocks_resolved': dialogue['all_blocks_resolved'],
                'resolved_blocks': ','.join(dialogue['resolved_blocks']),
                'avg_blocks_per_turn': sum(dialogue['blocks_resolved_per_turn']) / dialogue['total_turns'],
                'friction_scores':dialogue['friction_scores']
            }
            dialogue_rows.append(dialogue_row)
            
            # Create turn-level rows

            # print("turn metrics", dialogue)
            for turn in dialogue['turn_metrics']:
                turn_row = {
                    'model': model_name,
                    'dialogue_id': dialogue_id,
                    'turn_number': turn['turn_number'],
                    'blocks_resolved': turn['blocks_resolved'],
                    'total_blocks_resolved': turn['total_blocks_resolved'],
                    'resolved_blocks': ','.join(turn.get('resolved_blocks', [])),
                    'parsed_friction': turn.get('parsed_friction', ''),
                    'gpt_friction': turn.get('gpt_friction', ''),
                    'turn_level_friction_score': turn['turn_level_friction_score']

                }
                
                # Add quality metrics
                if 'quality' in turn:
                    for metric, value in turn['quality'].items():
                        turn_row[f'quality_{metric}'] = value
                
                # Add persuasiveness metrics
                if 'persuasiveness' in turn:
                    for metric, value in turn['persuasiveness'].items():
                        turn_row[f'persuasiveness_{metric}'] = value

                if 'turn_level_friction_score' in turn:
                    turn_row['friction_score'] = turn['turn_level_friction_score']

                
                
                turn_rows.append(turn_row)
        
        # Create dataframes and save as CSV for this model
        dialogue_df = pd.DataFrame(dialogue_rows)
        turn_df = pd.DataFrame(turn_rows)
        
        model_name_safe = model_name.replace('/', '_')
        dialogue_df.to_csv(f"{output_dir}/{model_name_safe}_dialogue_metrics_{timestamp}.csv", index=False)
        turn_df.to_csv(f"{output_dir}/{model_name_safe}_turn_metrics_{timestamp}.csv", index=False)
        
        print(f"\nResults for {model_name} saved to:")
        print(f"Summary metrics: {model_output_summary_file}")
        print(f"Detailed metrics: {model_output_detailed_file}")
        print(f"Dialogue metrics CSV: {output_dir}/{model_name_safe}_dialogue_metrics_{timestamp}.csv")
        print(f"Turn metrics CSV: {output_dir}/{model_name_safe}_turn_metrics_{timestamp}.csv")
    
    # Print comparative metrics across models
    print("\n\n===== COMPARATIVE METRICS ACROSS MODELS =====\n")
    
    # Create a comparative CSV file
    comparative_rows = []
    
    for metric_category in ['quality', 'efficiency', 'persuasiveness', 'friction_scores']:  # Add friction_scores
        print(f"\n--- {metric_category.upper()} METRICS ---")
    
        for metric_name, model_values in comparative_metrics[metric_category].items():
            # Skip if no values
            if not model_values:
                continue
                
            # Print the comparative metrics
            print(f"\n{metric_name}:")
            for model_name, value in model_values.items():
                print(f"  {model_name}: {value:.4f}")
            
            # Identify best model for this metric
            best_model = max(model_values.items(), key=lambda x: x[1])
            print(f"  Best model: {best_model[0]} ({best_model[1]:.4f})")
            
            # Add to comparative rows for CSV
            for model_name, value in model_values.items():
                comparative_rows.append({
                    'category': metric_category,
                    'metric': metric_name,
                    'model': model_name,
                    'value': value,
                    'is_best': model_name == best_model[0]
                })
        
    # Save comparative metrics as CSV
    # print("comparative_rows", comparative_rows)
    comparative_df = pd.DataFrame(comparative_rows)
    comparative_csv = f"{output_dir}/comparative_metrics_{timestamp}.csv"
    comparative_df.to_csv(comparative_csv, index=False)
    
    # Save combined metrics for all models
    all_models_output_file = f"{output_dir}/all_models_metrics_{timestamp}.json"
    with open(all_models_output_file, 'w') as f:
        json.dump(all_models_metrics, f, indent=2)
    
    # Create a combined dialogue metrics CSV with model as a column
    all_dialogue_rows = []
    all_turn_rows = []
    
    for model_name, metrics in all_models_metrics.items():
        for dialogue in metrics['dialogues']:
            dialogue_id = dialogue['dialogue_id']
            
            # Create dialogue-level row
            dialogue_row = {
                'model': model_name,
                'dialogue_id': dialogue_id,
                'total_turns': dialogue['total_turns'],
                'turns_until_resolution': dialogue['turns_until_resolution'],
                'resolution_rate': dialogue['resolution_rate'],
                'final_blocks_resolved': dialogue['final_blocks_resolved'],
                'all_blocks_resolved': dialogue['all_blocks_resolved'],
                'resolved_blocks': ','.join(dialogue['resolved_blocks']),
                'avg_blocks_per_turn': sum(dialogue['blocks_resolved_per_turn']) / dialogue['total_turns']
            }
            all_dialogue_rows.append(dialogue_row)
            
            # Create turn-level rows
            for turn in dialogue['turn_metrics']:
                turn_row = {
                    'model': model_name,
                    'dialogue_id': dialogue_id,
                    'turn_number': turn['turn_number'],
                    'blocks_resolved': turn['blocks_resolved'],
                    'total_blocks_resolved': turn['total_blocks_resolved'],
                    'resolved_blocks': ','.join(turn.get('resolved_blocks', [])),
                    'parsed_friction': turn.get('parsed_friction', ''),
                    'gpt_friction': turn.get('gpt_friction', ''),
                    'turn_level_friction_score': turn['turn_level_friction_score']

                }
                if 'turn_level_friction_score' in turn:
                    turn_row['friction_score'] = turn['turn_level_friction_score']
                # Add quality metrics
                if 'quality' in turn:
                    for metric, value in turn['quality'].items():
                        turn_row[f'quality_{metric}'] = value
                
                # Add persuasiveness metrics
                if 'persuasiveness' in turn:
                    for metric, value in turn['persuasiveness'].items():
                        turn_row[f'persuasiveness_{metric}'] = value
                
                all_turn_rows.append(turn_row)
    
    # Create combined dataframes and save as CSV
    all_dialogue_df = pd.DataFrame(all_dialogue_rows)
    all_turn_df = pd.DataFrame(all_turn_rows)
    
    all_dialogue_csv = f"{output_dir}/all_models_dialogue_metrics_{timestamp}.csv"
    all_turn_csv = f"{output_dir}/all_models_turn_metrics_{timestamp}.csv"
    
    all_dialogue_df.to_csv(all_dialogue_csv, index=False)
    all_turn_df.to_csv(all_turn_csv, index=False)
    
    print(f"\n\nCombined results saved to:")
    print(f"All models metrics: {all_models_output_file}")
    print(f"Comparative metrics CSV: {comparative_csv}")
    print(f"Combined dialogue metrics CSV: {all_dialogue_csv}")
    print(f"Combined turn metrics CSV: {all_turn_csv}")

    
    # Create pivot tables for easier analysis
    # print("comparative_rows", comparative_rows)


    try:
        comparative_rows_with_std = []
        
        # Create a DataFrame from comparative_rows to make manipulation easier
        comparative_df = pd.DataFrame(comparative_rows)
        
        # For each row in comparative_rows, we'll add both mean and std versions
        for index, row in comparative_df.iterrows():
            # Add the mean row (already has the values)
            mean_row = row.copy()
            mean_row['value_type'] = 'mean'
            comparative_rows_with_std.append(mean_row.to_dict())
            
            # Create std row with defaults
            std_row = row.copy()
            std_row['value_type'] = 'std'
            std_row['is_best'] = False
            # print("all_models_metrics", row)
            
            # Try to get std value if possible, otherwise leave as NaN/None
            try:
                model = row['model']
                category = row['category']
                metric = row['metric']
                
                # Skip trying to look up std values for now - just set to None
                std_row['value'] = None
            except:
                std_row['value'] = None
            
            comparative_rows_with_std.append(std_row.to_dict())
        
        # Create DataFrame with both means and stds
        comparative_df_with_std = pd.DataFrame(comparative_rows_with_std)
        
        # Create pivot table that includes both means and standard deviations
        pivot_df = comparative_df_with_std.pivot(
            index='model', 
            columns=['category', 'metric', 'value_type'], 
            values='value'
        )
        pivot_csv = f"{output_dir}/model_metrics_pivot_{timestamp}.csv"
        pivot_df.to_csv(pivot_csv)
        
        # Similarly for dialogue metrics (this shouldn't be affected)
        dialogue_pivot = all_dialogue_df.pivot_table(
            index='model',
            values=['resolution_rate', 'turns_until_resolution', 'all_blocks_resolved'],
            aggfunc={'resolution_rate': ['mean', 'std'], 
                    'turns_until_resolution': ['mean', 'std'], 
                    'all_blocks_resolved': ['mean', 'std']}
        )
        dialogue_pivot_csv = f"{output_dir}/dialogue_success_by_model_{timestamp}.csv"
        dialogue_pivot.to_csv(dialogue_pivot_csv)
        
    except Exception as e:
        print(f"Error creating pivot tables: {str(e)}")
        import traceback
        traceback.print_exc()
 

    return all_models_metrics


def plot_model_comparison(all_models_metrics, output_dir=None, timestamp=None):
    """
    Plot performance comparisons across all models for various metrics.
    Creates a single large figure with subplots, using Seaborn styling.
    
    Args:
        all_models_metrics (dict): Dictionary with metrics for all models
        output_dir (str, optional): Directory to save plot files
        timestamp (str, optional): Timestamp for file naming
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from datetime import datetime
    import pandas as pd
    import seaborn as sns
    import math
    
    # Set default values if not provided
    if output_dir is None:
        output_dir = "friction_role_play_evaluation_results"
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp = ''
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set Seaborn style
    sns.set_theme(style="darkgrid", palette="deep")
    plt.rcParams['figure.facecolor'] = '#F0F0F0'  # Light grey background
    plt.rcParams['axes.facecolor'] = '#F8F8F8'    # Slightly lighter grey for plot area
    
    # Extract model names
    model_names = list(all_models_metrics.keys())
    model_display_names = [os.path.basename(model) for model in model_names]
    
    # Set color palette
    colors = sns.color_palette("deep", len(model_names))
    
    # Process each category of metrics (now including friction_scores)
    for category in ['quality', 'efficiency', 'persuasiveness', 'friction_scores']:
        # Get all metrics in this category across all models
        all_metrics = set()
        for model in model_names:
            if category in all_models_metrics[model]:
                # Special handling for friction_scores which might have a different structure
                if category == 'friction_scores' and isinstance(all_models_metrics[model][category], dict) and 'mean' in all_models_metrics[model][category]:
                    all_metrics.add('friction_score')
                else:
                    all_metrics.update(all_models_metrics[model][category].keys())
        
        # Convert to sorted list for consistent ordering
        all_metrics = sorted(list(all_metrics))
        
        # Skip if no metrics found for this category
        if not all_metrics:
            print(f"No metrics found for category: {category}")
            continue
        
        # Calculate grid dimensions
        num_metrics = len(all_metrics)
        num_cols = 4  # Four plots per row
        num_rows = math.ceil(num_metrics / num_cols)
        
        # Create figure and subplots
        fig = plt.figure(figsize=(5 * num_cols, 4 * num_rows))
        fig.suptitle(f'{category.capitalize()} Metrics Comparison', fontsize=24, y=0.98)
        
        # Create subplots for each metric
        for i, metric in enumerate(all_metrics):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            
            # Extract data for this metric across all models
            metric_values = []
            metric_errors = []
            
            for model in model_names:
                # Handle different structures for different categories
                if category in all_models_metrics[model]:
                    if category == 'friction_scores' and isinstance(all_models_metrics[model][category], dict) and 'mean' in all_models_metrics[model][category]:
                        # Direct access for friction_scores
                        stat = all_models_metrics[model][category]
                        metric_values.append(stat['mean'])
                        metric_errors.append(stat['std'])
                    elif metric in all_models_metrics[model][category]:
                        # Standard access for other metrics
                        stat = all_models_metrics[model][category][metric]
                        metric_values.append(stat['mean'])
                        metric_errors.append(stat['std'])
                    else:
                        metric_values.append(0)
                        metric_errors.append(0)
                else:
                    metric_values.append(0)
                    metric_errors.append(0)
            
            # Create a bar plot for this metric
            bars = ax.bar(model_display_names, metric_values, yerr=metric_errors, 
                    capsize=5, color=colors, alpha=0.8)
            
            # Add value labels above bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{metric_values[j]:.2f}',
                        ha='center', va='bottom', rotation=0,
                        fontsize=8)
            
            # Set subplot title and labels
            ax.set_title(metric, fontsize=12)
            ax.set_ylabel(f'Value', fontsize=10)
            
            # Only show x-axis labels for bottom row or last plot in a column
            is_bottom_row = (i // num_cols) == (num_rows - 1) 
            is_last_plot_of_partial_row = (i == len(all_metrics) - 1)
            
            if is_bottom_row or is_last_plot_of_partial_row:
                # Rotate x-axis labels for readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            else:
                # Hide x-axis tick labels for non-bottom rows
                ax.set_xticklabels([])
            
            # Adjust y-axis to add some space for labels
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min, y_max * 1.15)
        
        # Adjust layout and save
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
        plt.savefig(f"{output_dir}/{category}_metrics_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create block resolution metrics comparison
    # Extract block resolution data
    resolution_data = {}
    turn_data = {}
    
    for i, model in enumerate(model_names):
        resolution_data[model] = {}
        turn_data[model] = {}
        
        # Get block metrics for this model
        block_metrics = all_models_metrics[model]['block_difficulty']['block_metrics']
        
        for block, metrics in block_metrics.items():
            resolution_data[model][block] = metrics['resolution_rate']
            # Handle None values for avg_turn_to_resolve
            if metrics['avg_turn_to_resolve'] is not None:
                turn_data[model][block] = metrics['avg_turn_to_resolve']
            else:
                turn_data[model][block] = np.nan  # Use NaN for plotting
    
    # Convert to dataframes for easier plotting
    resolution_df = pd.DataFrame(resolution_data)
    turn_df = pd.DataFrame(turn_data)
    
    # Create figure for block metrics
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Block Resolution Metrics', fontsize=24, y=0.98)
    
    # Plot block resolution rates
    resolution_df.plot(kind='bar', ax=axes[0], color=colors, alpha=0.8)
    axes[0].set_title('Block Resolution Rate by Model', fontsize=14)
    axes[0].set_xlabel('Block', fontsize=12)
    axes[0].set_ylabel('Resolution Rate', fontsize=12)
    axes[0].legend(model_display_names)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot average turns to resolve
    turn_df.plot(kind='bar', ax=axes[1], color=colors, alpha=0.8)
    axes[1].set_title('Average Turns to Resolve Block by Model', fontsize=14)
    axes[1].set_xlabel('Block', fontsize=12)
    axes[1].set_ylabel('Average Turns', fontsize=12)
    axes[1].legend(model_display_names)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.savefig(f"{output_dir}/block_resolution_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create efficiency summary plot
    efficiency_metrics = ['turns_until_resolution', 'resolution_rate', 'avg_blocks_per_turn']
    efficiency_values = {model: [] for model in model_names}
    
    for model in model_names:
        for metric in efficiency_metrics:
            if metric in all_models_metrics[model]['efficiency']:
                efficiency_values[model].append(all_models_metrics[model]['efficiency'][metric]['mean'])
            else:
                efficiency_values[model].append(0)
    
    # Create efficiency plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Efficiency Metrics Comparison', fontsize=20)
    
    # Create grouped bar chart
    x = np.arange(len(efficiency_metrics))
    width = 0.8 / len(model_names)
    
    for i, model in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, efficiency_values[model], width, 
                    label=model_display_names[i], color=colors[i], alpha=0.8)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{efficiency_values[model][j]:.2f}',
                    ha='center', va='bottom', rotation=0,
                    fontsize=8)
    
    ax.set_xlabel('Efficiency Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(efficiency_metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
    plt.savefig(f"{output_dir}/efficiency_summary_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create radar chart for comprehensive model comparison
    common_metrics = {}
    for category in ['quality', 'efficiency', 'persuasiveness']:
        common_metrics[category] = set.intersection(
            *[set(all_models_metrics[model][category].keys()) for model in model_names]
        )
    
    # Flatten into a single list of metrics
    all_common_metrics = []
    for category, metrics in common_metrics.items():
        for metric in metrics:
            all_common_metrics.append(f"{category}_{metric}")
    
    # Only proceed if we have common metrics
    if all_common_metrics:
        # Prepare data for radar chart
        radar_data = {model: [] for model in model_display_names}
        
        for model, display_name in zip(model_names, model_display_names):
            for category in ['quality', 'efficiency', 'persuasiveness']:
                for metric in common_metrics[category]:
                    radar_data[display_name].append(all_models_metrics[model][category][metric]['mean'])
        
        # Normalize data for radar chart (0-1 scale)
        radar_df = pd.DataFrame(radar_data, index=all_common_metrics)
        radar_df_norm = radar_df.copy()
        
        for metric in all_common_metrics:
            min_val = radar_df.loc[metric].min()
            max_val = radar_df.loc[metric].max()
            if max_val > min_val:
                radar_df_norm.loc[metric] = (radar_df.loc[metric] - min_val) / (max_val - min_val)
            else:
                radar_df_norm.loc[metric] = radar_df.loc[metric] / max_val if max_val != 0 else 0
        
        # Create radar chart
        fig = plt.figure(figsize=(12, 10), facecolor='#F0F0F0')
        fig.suptitle('Model Comparison Across All Metrics (Normalized)', fontsize=20)
        
        # Number of variables
        N = len(all_common_metrics)
        
        # Create angles for radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create subplot with polar projection
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor('#F8F8F8')
        
        # Add metric labels
        plt.xticks(angles[:-1], all_common_metrics, size=8)
        
        # Draw the outline of the chart
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)
        
        # Plot each model
        for i, model in enumerate(model_display_names):
            values = radar_df_norm[model].values.flatten().tolist()
            values += values[:1]  # Close the loop
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        plt.savefig(f"{output_dir}/radar_chart_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"All comparison plots saved to {output_dir}/")

def sample_unique_group_ids(dataset, n=100, seed=42):
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Get all unique group IDs from the dataset
    unique_group_ids = list(set(dataset['group_id']))
    
    # Sample n unique group IDs (or all if there are fewer than n)
    n_sample = min(n, len(unique_group_ids))
    sampled_ids = random.sample(unique_group_ids, n_sample)
    
    return sampled_ids


def truncate_dialogue_history(dialogue_history, tokenizer, max_tokens=6000):
    """
    Truncates dialogue history by keeping the most recent context around the last two
    Friction Agent interactions.
    
    Args:
        dialogue_history (str): The full dialogue history
        tokenizer: The tokenizer to use for counting tokens
        max_tokens (int): Maximum number of tokens to keep
        
    Returns:
        str: Truncated dialogue history
    """
    # Find the positions of all "Friction Agent:" occurrences
    friction_positions = []
    pos = 0
    
    while True:
        pos = dialogue_history.find("Friction Agent:", pos)
        if pos == -1:
            break
        friction_positions.append(pos)
        pos += 1
    
    # If we have fewer than 2 friction agent occurrences, return the original
    if len(friction_positions) < 2:
        return dialogue_history
    
    # Get the position of the second-to-last friction agent occurrence
    truncate_pos = friction_positions[-2]
    
    # Get the truncated history
    truncated_history = dialogue_history[truncate_pos:]
    
    # Check if the truncated history is still too long
    encoded = tokenizer.encode(truncated_history)
    if len(encoded) <= max_tokens:
        return truncated_history
    
    # If it's still too long, keep only the last friction agent interaction
    if len(friction_positions) >= 1:
        truncate_pos = friction_positions[-1]
        truncated_history = dialogue_history[truncate_pos:]
        
        encoded = tokenizer.encode(truncated_history)
        if len(encoded) <= max_tokens:
            return truncated_history
    
    # If it's still too long, do a hard truncation
    encoded = tokenizer.encode(dialogue_history)
    if len(encoded) > max_tokens:
        # Keep the last max_tokens tokens
        truncated_tokens = encoded[-max_tokens:]
        # Decode back to text
        truncated_history = tokenizer.decode(truncated_tokens)
        return truncated_history
    
    return dialogue_history


def add_initial_solutions(combined_dataset, delidata_corpus):
    """Add initial solution data to the combined dataset."""
    # Create a dictionary to store initial solutions by group_id
    initial_solutions_by_group = {}
    
    
    # Extract initial solutions from delidata corpus
    for group_id, messages in delidata_corpus.corpus.items():
        if messages and 'sol_tracker_all' in messages[0]:
            initial_solutions_by_group[group_id] = messages[0]['sol_tracker_all']
    
    # Add the solutions to the combined dataset
    new_features = combined_dataset.features.copy()
    new_features['initial_solutions'] = {}  # Add the new field
    
    # Create new dataset with the additional field
    new_data = {k: [] for k in new_features}
    
    # Fill the new dataset
    for i in range(len(combined_dataset)):
        # Copy existing fields
        for k in combined_dataset.features:
            new_data[k].append(combined_dataset[i][k])
        
        # Add initial solutions
        group_id = combined_dataset[i]['group_id']
        if group_id in initial_solutions_by_group:
            new_data['initial_solutions'].append(initial_solutions_by_group[group_id])
        else:
            new_data['initial_solutions'].append({})  # Empty dict if no data
    
    # Create and return the new dataset
    from datasets import Dataset
    return Dataset.from_dict(new_data)

def check_stopping_condition(parsed_gpt):
    """
    Checks if the group has reached a complete decision on all cards.
    
    Args:
        parsed_gpt (dict): Parsed GPT completion
        
    Returns:
        bool: True if stopping condition is met, False otherwise
    """
    # Check if there's a final submission
    if "final_submission" in parsed_gpt:
        return True
        
    # Check common ground - if all 4 cards have clear status
    if "common_ground" in parsed_gpt:
        card_identifiers = set()
        for item in parsed_gpt["common_ground"]:
            if "card" in item:
                card_identifiers.add(item["card"])
        
        # If all cards from the task are in common ground
        if len(card_identifiers) >= 4:
            return True
            
    # Check if under_debate is empty and common_ground has items
    if ("under_debate" in parsed_gpt and not parsed_gpt["under_debate"] and 
        "common_ground" in parsed_gpt and parsed_gpt["common_ground"]):
        return True
        
    return False


def process_dialogues(
    models_list,
    test_dialog_id_list,
    use_chat_completion,
    combined_dataset,
    tokenizer_base_path,
    output_dir="friction_roleplay_evals",
    max_turns=10,
    generation_args=None,
    chat_client=None,
    gpt_model_name="gpt-4o-mini",
    seed=42,
    reward_model=None, 
    best_of_n=None, 
    top_k_candidates=1, 
    rm_tokenizer=None, 
    rm_max_length=None,
    use_adversarial_modification=False,
    adversarial_model_path=None,
    personality_combinations=None
):
    """
    Acutal roleplay loop starts here: essentially a two-way back and forth between two models: friction agent (model.generate) and collaborator (GPT)
    Process dialogues using multiple models, with the option to use either model.generate or chat completion.
    Adapted for DELI dataset with card selection task dialogues.
    
    Args:
        models_list (list): List of model paths/names to iterate through: trained friction agents;  baselines like DPO, PPO 
        test_dialog_id_list (list): List of dialogue IDs to process
        use_chat_completion (bool): Whether to use chat completion instead of model.generate
        combined_dataset (Dataset): DELI dataset containing dialogues
        tokenizer_base_path (str): Path to load tokenizer from
        output_dir (str): Directory to save results
        max_turns (int): Maximum number of dialogue turns
        generation_args (dict): Arguments for model generation
        chat_client: Client for chat completion API calls
        gpt_model_name (str): Name of the GPT model to use for chat completion
        seed (int): Random seed for reproducibility
        use_adversarial_modification (bool): Whether to use adversarial modification for GPT prompts
        adversarial_model_path (str): Path to the adversarial modification model
        
    Returns:
        dict: All conversations organized by model
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Set default generation args if not provided
    if generation_args is None:
        generation_args = {
            "max_new_tokens": 256,
            "temperature": 0.9,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.9,
            "num_beams": 5,
            "min_length": 100,
            'num_return_sequences': 1
        }
    
    # Initialize data structures to store results
    all_models_conversations = {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Current timestamp for filenames
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the main log file
    main_log_filename = f"{output_dir}/all_models_dialogue_generation_{current_time}.json"
    
    # Create the main markdown file
    main_md_filename = f"{output_dir}/all_models_dialogues_readable_{current_time}.md"
    
    # Initialize the markdown file with a header
    with open(main_md_filename, 'w') as f:
        f.write(f"# All Models Dialogue Generation Results - {current_time}\n\n")

    # Use the provided test_dialog_id_list (already contains group_ids to process)
    test_dialog_id_list = [x for x in combined_dataset['group_id']]
    print("FINAL TARGET ID LIST", len(test_dialog_id_list))
 
    # Standardize the personality_facets list
    personality_facets = personality_combinations
    print(personality_facets[0:5])

    # Standardize the personality_facets list first
    standardized_facets = []
    for personality in personality_facets:
        trait, facet = personality.split(':')
        trait = trait.capitalize()
        facet = facet.replace('_', ' ').strip()
        standardized_facets.append(f"{trait}:{facet}")

    # Load adversarial model if needed
    adversarial_model = None
    if use_adversarial_modification and not use_chat_completion and adversarial_model_path:
        print(f"Loading adversarial modification model from {adversarial_model_path}...")

        # Load the model
        adversarial_lora_model = AutoPeftModelForCausalLM.from_pretrained(
            adversarial_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Merge the model
        print("Merging LoRA adapter...")
        adversarial_model = adversarial_lora_model.merge_and_unload()
        print("Adversarial model merged", adversarial_model)
        
        # Load tokenizer
        adversarial_tokenizer = AutoTokenizer.from_pretrained(adversarial_model_path)
        adversarial_tokenizer.pad_token = "<|reserved_special_token_0|>"
        adversarial_tokenizer.padding_side = "right"

    # Loop through models
    for model_name in models_list:
        print("model_name", model_name)
        loading_model_name = model_name
        print("loading_model_name", loading_model_name)

        if "/" in model_name:
            parts = model_name.split("/")
            if len(parts) >= 2:
                # Combine the first two parts
                model_name = parts[0] + "_" + parts[1]
        if best_of_n:
            model_name = model_name + f"best_of_{best_of_n}"
            
        print(f"\n===== Processing Model: {model_name} =====\n")
        
        # Create model-specific log files
        model_log_filename = f"{output_dir}/dialogue_generation_log_{os.path.basename(model_name)}_{current_time}.json"
        model_md_filename = f"{output_dir}/dialogues_readable_{os.path.basename(model_name)}_{current_time}.md"
        
        # Initialize the model-specific markdown file
        with open(model_md_filename, 'w') as f:
            f.write(f"# Dialogue Generation Results for {model_name} - {current_time}\n\n")
        
        # Initialize data for this model
        all_conversations = {}
        
        # Load model and tokenizer if not using chat completion
        if not use_chat_completion:
            print(f"Loading model from {model_name}...")

            # Load the model
            lora_model = AutoPeftModelForCausalLM.from_pretrained(
                loading_model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            
            # Merge the model
            print("Merging LoRA adapter...")
            merged_model = lora_model.merge_and_unload()
            print("MERGED LORA FRICTION AGENT")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(loading_model_name)
            tokenizer.pad_token = "<|reserved_special_token_0|>"
            tokenizer.padding_side = "right"
        else:
            print(f"Using chat completion with model {gpt_model_name}...")
            merged_model = None
            tokenizer = None
        
        # Calculate save frequency
        save_frequency = max(1, len(test_dialog_id_list) // 3)
        print(f"Will save results after every {save_frequency} dialogues processed")
        
        # Counter for processed dialogues
        processed_count = 0
        print("SIZE of DUMMY DELI data", len(combined_dataset), combined_dataset)
        
        # Main loop to iterate through dialogues
        for index, entry in enumerate(tqdm(combined_dataset, desc=f"Processing dialogues with {model_name}")):
            group_id = entry['group_id']
            
            # Skip if we've already processed this ID
            if group_id in all_conversations:
                continue
            
            # Process only IDs in our target list
            if group_id in test_dialog_id_list:
                print(f"Processing dialogue ID: {group_id}")
                
                # Extract participants from context
                participants = []
                if '&&' in entry['context']:
                    participant_section = entry['context'].split('&&')[0]
                    raw_participants = participant_section.replace('SYSTEM:', '').strip().split(',')
                    participants = [p.strip() for p in raw_participants if p.strip() != 'SYSTEM']
                
                # Sample random personalities for participants
                n_participants = len(participants)
                sampled_personalities = random.sample(standardized_facets, n_participants)

                initial_solutions_raw = entry.get('initial_solutions', '{}')
                print("initial initial_solutions_raw", initial_solutions_raw)
                # Check if it's a string and parse it if needed
                if isinstance(initial_solutions_raw, str):
                    import json
                    try:
                        initial_solutions = json.loads(initial_solutions_raw)
                    except json.JSONDecodeError:
                        # If parsing fails, use an empty dictionary
                        print(f"Warning: Could not parse initial_solutions: {initial_solutions_raw}")
                        initial_solutions = {}
                else:
                    initial_solutions = initial_solutions_raw

                # Create combined personality and solution information
                print("initial initial_solutions", initial_solutions)
                personalities_with_solutions = {}
                for i, p in enumerate(participants):
                    # Extract the personality and facet properly
                    personality_facet = sampled_personalities[i]  # This already has the format "Trait:Facet"
                    
                    # Get initial cards selected by this participant
                    initial_cards = initial_solutions.get(p, [])
                    solution_str = ", ".join(initial_cards) if initial_cards else "none"
                    
                    # Combine personality with initial solution
                    personalities_with_solutions[p] = f"{personality_facet} (Initial selection: {solution_str})"
                    
                # Initialize conversation record
                conversation_record = {
                    'dialog_id': group_id,
                    'original_context': entry['context'],
                    'gold_friction_bootstrap': entry['chosen_friction'],
                    'belief_state': entry.get('belief_state', ''),
                    'contradiction_reason': entry.get('contradiction_reason', ''),
                    'personalities': personalities_with_solutions,
                    'initial_solutions': initial_solutions,  # Also store the raw data
                    'use_adversarial_modification': use_adversarial_modification,
                    'turns': []
                }
                
                print("Context:", entry['context'])
                print("Friction GOLD:", entry['chosen_friction'])
                
                # Initialize the dialogue history with the original context
                current_dialogue_history = entry['context']
                
                # Main conversation generation loop
                for turn in range(max_turns):
                    print(f"\n----- TURN {turn+1} -----")
                    turn_data = {
                        'turn_number': turn + 1,
                        'dialogue_before_friction': current_dialogue_history,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # STEP 1: Generate friction with model
                    policy_inputs = {}
                    policy_inputs['context'] = current_dialogue_history

                    # Truncate the dialogue history if it's getting too long
                    if len(current_dialogue_history.split()) > 500:
                        if not use_chat_completion:
                            original_length = len(tokenizer.encode(current_dialogue_history))
                            print(f"[DEBUG] Original dialogue history: {original_length} tokens, {len(current_dialogue_history.split())} words")
                            policy_inputs['context'] = truncate_dialogue_history(
                                policy_inputs['context'], 
                                tokenizer, 
                                max_tokens=6000
                            )
                            truncated_length = len(tokenizer.encode(policy_inputs['context']))
                            print(f"[DEBUG] After truncation: {truncated_length} tokens ({(truncated_length/original_length)*100:.1f}% of original)")

                    # Choose the appropriate template
                    if not use_chat_completion:
                        policy_inputs.update(process_data_template(policy_inputs))
                    else:
                        policy_inputs.update(process_data_template_chat_completion(policy_inputs))
                    
                    print(f"Generating friction for turn {turn+1}...")
                    
                    try:
                        # Choose between model.generate and chat completion
                        if use_chat_completion:
                            # Using chat completion
                            system_prompt_rm = '''You are an expert in collaborative task analysis and reasoning.  
                                    Participants must test the rule: **"All cards with vowels have an even number on the other side."**  
                                    Generate a friction intervention that fosters self-reflection, realigns understanding, and supports collaboration.  
 
                                    <rationale>  
                                    Explain why intervention is needed—what's misaligned, its impact, and the expected resolution.  
                                    </rationale>  

                                    <friction>  
                                    Generate a friction intervention that fosters self-reflection, realigns understanding, and supports collaboration.  
                                    </friction>  
                                    '''
                            completion = chat_client.chat.completions.create(
                                model=gpt_model_name,
                                messages=[
                                    {"role": "system", "content": system_prompt_rm},
                                    {"role": "user", "content": policy_inputs['prompt']}
                                ]
                            )
                            
                            if completion and completion.choices:
                                text_to_parse = completion.choices[0].message.content
                                turn_data['model_generated_text'] = text_to_parse
                            else:
                                print("Failed to get chat completion")
                                turn_data['generation_error'] = "Empty or invalid chat completion"
                                conversation_record['turns'].append(turn_data)
                                break
                        else:
                            # Using model.generate
                            if best_of_n:
                                print("Running BON")
                                device = 'cuda:0'

                                generated_texts, all_generated_texts = generate_multiple_sequences_with_intrinsic_metrics(
                                    merged_model, 
                                    tokenizer, 
                                    policy_inputs['prompt'], 
                                    generation_args, 
                                    device = device,
                                    strategy="best_of_n", 
                                    batched=True,reward_model=reward_model, best_of_n=best_of_n, 
                                    top_k_candidates=top_k_candidates, rm_tokenizer = rm_tokenizer, rm_max_length = rm_max_length
                                )
                            else:
                                generated_texts, all_generated_texts = generate_multiple_sequences_with_intrinsic_metrics(
                                    merged_model, 
                                    tokenizer, 
                                    policy_inputs['prompt'], 
                                    generation_args, 
                                    None,
                                    strategy="top_p_sampling", 
                                    batched=True
                                )
                        
                            # Process the generated text
                            if generated_texts and isinstance(generated_texts, list):
                                text_to_parse = generated_texts[0][0] if (generated_texts[0] and isinstance(generated_texts[0], list)) else generated_texts[0]
                                turn_data['model_generated_text'] = text_to_parse
                            else:
                                print("Failed to generate text")
                                turn_data['generation_error'] = "Empty or invalid generated text"
                                conversation_record['turns'].append(turn_data)
                                break
                        
                        # Parse tags from generated text
                        parsed_frictive_states_and_friction = parse_tags_robust(text_to_parse, tags_for_parsing)
                        
                        # Extract components
                        friction_intervention = ' '.join(parsed_frictive_states_and_friction.get('friction', []))
                        if not friction_intervention:
                            friction_intervention = handle_friction_logic(text_to_parse)
                        
                        task_state = ' '.join(parsed_frictive_states_and_friction.get('t', []))
                        belief_state = ' '.join(parsed_frictive_states_and_friction.get('b', []))
                        rationale = ' '.join(parsed_frictive_states_and_friction.get('rationale', []))
                        
                        # Store in turn data
                        turn_data.update({
                            'parsed_friction': friction_intervention,
                            'task_state': task_state,
                            'belief_state': belief_state,
                            'rationale': rationale
                        })
                        
                        print(f"Generated friction: {friction_intervention}")
                    except Exception as e:
                        print(f"Error in generation: {str(e)}")
                        turn_data['generation_error'] = str(e)
                        conversation_record['turns'].append(turn_data)
                        break
                    
                    # STEP 2: Add friction to dialogue and prepare GPT prompt
                    updated_dialogue = current_dialogue_history + "\nFriction Agent: " + friction_intervention
                    turn_data['dialogue_with_friction'] = updated_dialogue
                    if not use_adversarial_modification:
                        # Format standard GPT prompt with updated dialogue
                        if turn == 0:
                            # Create a formatted string of personalities with solutions
                            personality_str = ""
                            for participant, combined_info in personalities_with_solutions.items():
                                personality_str += f"{participant}: {combined_info}\n"

                            # Add card information to the prompt
                            cards_info = entry.get('task_cards', '')
                            if cards_info:
                                cards_str = f"{cards_info}]\n"
                            else:
                                cards_str = ""
                                                    
                            standard_gpt_prompt = gpt_user_continuation_prompt_bootstrap.format(
                                dialogue=updated_dialogue,
                                personalities=personality_str,
                                cards_info=cards_str 
                            )

                        elif turn > 0 and turn < max_turns - 1:
                            personality_str = ""
                            for participant, combined_info in personalities_with_solutions.items():
                                personality_str += f"{participant}: {combined_info}\n"
                            
                            cards_info = entry.get('task_cards', '')
                            if cards_info:
                                cards_str = f"{cards_info}]\n"
                            else:
                                cards_str = ""
    
                            standard_gpt_prompt = gpt_user_continuation_prompt_onwards.format(
                                dialogue=updated_dialogue,
                                personalities=personality_str,
                                cards_info=cards_str 
                            )

                        
                        else:
                            # Create a formatted string of personalities with solutions
                            personality_str = ""
                            for participant, combined_info in personalities_with_solutions.items():
                                personality_str += f"{participant}: {combined_info}\n"
                                
                            cards_info = entry.get('task_cards', '')
                            if cards_info:
                                cards_str = f"{cards_info}]\n"
                            else:
                                cards_str = ""
                            standard_gpt_prompt = gpt_user_continuation_prompt_final.format(
                                dialogue=updated_dialogue,
                                personalities=personality_str,
                                cards_info=cards_str 
                            )

                        turn_data['standard_gpt_prompt'] = standard_gpt_prompt
                    
                    # STEP 3: Get GPT response with potential adversarial modification
                    if use_adversarial_modification:
                        print(f"Calling GPT for turn {turn+1}...")

                      
                        if turn == 0:
                            # Create a formatted string of personalities with solutions
                            personality_str = ""
                            for participant, combined_info in personalities_with_solutions.items():
                                personality_str += f"{participant}: {combined_info}\n"

                            # Add card information to the prompt
                            cards_info = entry.get('task_cards', '')
                            if cards_info:
                                cards_str = f"{cards_info}]\n"
                            else:
                                cards_str = ""
                                                    
                            standard_gpt_prompt = gpt_ma_adversarial_system_prompt.format(
                                dialogue=updated_dialogue,
                                personalities=personality_str,
                                cards_info=cards_str 
                            )

                        elif turn > 0 and turn < max_turns - 1:
                            personality_str = ""
                            for participant, combined_info in personalities_with_solutions.items():
                                personality_str += f"{participant}: {combined_info}\n"
                            
                            cards_info = entry.get('task_cards', '')
                            if cards_info:
                                cards_str = f"{cards_info}]\n"
                            else:
                                cards_str = ""
    
                            standard_gpt_prompt = gpt_user_continuation_prompt_onwards_ma_adversarial.format(
                                dialogue=updated_dialogue,
                                personalities=personality_str,
                                cards_info=cards_str 
                            )

                        
                        else:
                            # Create a formatted string of personalities with solutions
                            personality_str = ""
                            for participant, combined_info in personalities_with_solutions.items():
                                personality_str += f"{participant}: {combined_info}\n"
                                
                            cards_info = entry.get('task_cards', '')
                            if cards_info:
                                cards_str = f"{cards_info}]\n"
                            else:
                                cards_str = ""
                            # final GPT prompt, no adversarial action modification here
                            standard_gpt_prompt = gpt_user_continuation_prompt_final.format(
                                dialogue=updated_dialogue,
                                personalities=personality_str,
                                cards_info=cards_str 
                            )

                        turn_data['standard_gpt_prompt'] = standard_gpt_prompt
 
                    
                    # STEP 4: Update dialogue history for next turn
                    current_dialogue_history = updated_dialogue + "\n" + gpt_continuation
                    turn_data['updated_dialogue'] = current_dialogue_history
                    
                    # Add this turn's data to conversation record
                    conversation_record['turns'].append(turn_data)

                    # Check if we should mark the task as completed but continue the dialogue
                    if parsed_gpt and check_stopping_condition(parsed_gpt):
                        print(f"Task completed! But continuing for data collection...")
                        conversation_record['stopping_reason'] = "Task completed"
                        # No break or continue - we just mark it and keep going
                        
                    # When we reach max turns, add that as a reason if not already set
                    if turn == max_turns - 1 and 'stopping_reason' not in conversation_record:
                        conversation_record['stopping_reason'] = "Maximum turns reached"
                                    
                # Save the completed conversation to the overall record
                all_conversations[group_id] = conversation_record
                processed_count += 1
                
                # Append this dialogue to the model-specific markdown file
                with open(model_md_filename, 'a') as f:
                    f.write(f"\n## Dialogue {group_id}\n\n")
                    f.write(f"### Original Context\n\n```\n{entry['context']}\n```\n\n")
                    f.write(f"### Gold Friction Bootstrap (T=1)\n\n```\n{entry['chosen_friction']}\n```\n\n")
                    f.write(f"### Belief frictive State Phi Bootstrap\n\n```\n{entry.get('belief_state', 'N/A')}\n```\n\n")
                    f.write(f"### Contradiction Reason Bootstrap \n\n```\n{entry.get('contradiction_reason', 'N/A')}\n```\n\n")
                    
                    # Add participant personalities and initial solutions
                    f.write(f"### Participant Personalities and Initial Solutions\n\n")
                    for participant, info in conversation_record['personalities'].items():
                        f.write(f"- **{participant}**: {info}\n")
                    f.write("\n")
                    f.write(f"### Valid Cards \n\n```\n{entry.get('task_cards')}\n```\n\n")

                    # Note whether adversarial modification was used
                    f.write(f"### Using Adversarial Modification: {conversation_record['use_adversarial_modification']}\n\n")

                    for turn_idx, turn in enumerate(conversation_record['turns']):
                        f.write(f"### Turn {turn_idx + 1}\n\n")
                        
                        # Note if adversarial modification was used in this turn
                        if 'used_adversarial_modification' in turn and turn['used_adversarial_modification']:
                            f.write(f"**Used Adversarial Modification**: Yes\n\n")
                        
                        f.write(f"**Friction**: {turn.get('parsed_friction', 'N/A')}\n\n")
                        
                        if 'gpt_utterances' in turn:
                            f.write("**GPT Utterances**:\n\n")
                            for utterance in turn['gpt_utterances']:
                                f.write(f"- {utterance}\n")
                            f.write("\n")

                        # Add common ground information
                        if 'parsed_gpt_response' in turn and 'common_ground' in turn['parsed_gpt_response']:
                            f.write("**Common Ground**:\n")
                            for item in turn['parsed_gpt_response']['common_ground']:
                                f.write(f"- {item['card']} ({item['action']})\n")
                                
                        # Add under debate information
                        if 'parsed_gpt_response' in turn and 'under_debate' in turn['parsed_gpt_response']:
                            f.write("**Under Debate**:\n")
                            for item in turn['parsed_gpt_response']['under_debate']:
                                f.write(f"- {item['card']} ({item['supporters']})\n")
                    
                    f.write("---\n\n")  # Separator between dialogues
                
                # Periodic save of JSON
                if processed_count % save_frequency == 0 or processed_count == len(test_dialog_id_list):
                    with open(model_log_filename, 'w') as f:
                        json.dump(all_conversations, f, indent=2)
                    print(f"Saved results to {model_log_filename} after processing {processed_count}/{len(test_dialog_id_list)} dialogues")
                
                # Check if we've processed all target dialogues
                if len(all_conversations) == len(test_dialog_id_list):
                    print



            # Check if we've processed all target dialogues
            if len(all_conversations) == len(test_dialog_id_list):
                print("All target dialogues processed for this model. Moving to next model.")
                break
    
        # Store results for this model
        all_models_conversations[model_name] = all_conversations
        
        # Add a section for this model in the main markdown file
        with open(main_md_filename, 'a') as f:
            f.write(f"# Model: {model_name}\n\n")
            f.write(f"Processed {len(all_conversations)} dialogues\n\n")
            f.write(f"Detailed results in: {model_md_filename}\n\n")
            f.write("---\n\n")  # Separator between models
            
        # Save individual model results
        with open(model_log_filename, 'w') as f:
            json.dump(all_conversations, f, indent=2)
        print(f"Saved final results for model {model_name} to {model_log_filename}")

        # Clean up models to free memory
        if not use_chat_completion:
            try:
                del lora_model
                del merged_model
         
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            except Exception as e:
                print(f"Error cleaning up models: {str(e)}")
            
    # Save final combined results
    with open(main_log_filename, 'w') as f:
        json.dump(all_models_conversations, f, indent=2)
    print(f"Saved combined results for all models to {main_log_filename}")
    print(f"Summary of all model results saved to {main_md_filename}")

    return all_models_conversations, main_log_filename

if __name__ == "__main__":
    # Define arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Set seed for reproducibility
    set_seed(script_args.seed)

    #load model and tokenizer

     # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16


    #load the original friction data
    output_folder_friction_data = "gpt4o_complete_dialogs" # should exist
 
    dialogs = pickle.load(open(f"{output_folder_friction_data}/friction_ranking_rogue_6194_partial.pkl", "rb"))
    personality_combinations = []
    for index, (key, entry) in enumerate(tqdm(dialogs.items(), desc=f"Processing dialogues")):
        original_friction = entry['friction_data_original']
        target_id = original_friction['dialog_id']
    
    #     print(f"Processing dialogue ID: {target_id}")
        personality_combinations.append(original_friction['P1_personality_type'] + ":" + original_friction['P1_facet'])
        personality_combinations.append(original_friction['P2_personality_type'] + ":" + original_friction['P2_facet'])
        personality_combinations.append(original_friction['P3_personality_type'] + ":" + original_friction['P3_facet'])
    personality_combinations = list(set(personality_combinations))
    # Initialize conversation record

    bon_models_list = ["SFT"] # trained SFT model
   

    models_list_deli_weights =[]
    test_dialog_id_list = []
    # Generation arguments
    generation_args = {
        "max_new_tokens": 356,
        "temperature": 0.2,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9,
        "num_beams": 5,
        "min_length": 100,
        'num_return_sequences': 1,
        "sampling_strategy": "top_p_sampling"
    }

 
    output_dir_api  = "friction_roleplay_evalsdeli"
    plot_output_dir = "friction_roleplay_evals_scaling_testing_plots_2"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_output_dir, exist_ok=True)
    # Set seed for reproducibility
    seed = 42
    # create the delidata dialogue for testing and include initial solution

    combined_dataset_with_solutions = load_from_disk("combined_dataset_with_initsolutions_DELI")


    all_model_results, main_log_filenam = process_dialogues(
            models_list=models_list_deli_weights,
            test_dialog_id_list=test_dialog_id_list,
            use_chat_completion = True,
            combined_dataset = combined_dataset_with_solutions,
            tokenizer_base_path=models_list[0],
            output_dir=output_dir_api,
            max_turns=15,
            generation_args=generation_args,
            chat_client=client,
            gpt_model_name="gpt-4o",
            seed=42,
            reward_model=None, 
            best_of_n=False, 
            top_k_candidates=1, 
            rm_tokenizer=None, 
            rm_max_length=None,
            include_ma_adversarial_agent=False,  
            ma_adversarial_model_path=None,  # Path to adversarial agent model,
         
    personality_combinations = personality_combinations
            
        )



