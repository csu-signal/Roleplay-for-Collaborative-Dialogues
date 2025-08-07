import torch
import gc
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
from typing import Optional, List
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
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
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
        default="friction_sft_allsamples_weights_instruct",
        metadata={"help": "the location of the SFT model name or path"},
    )

    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    
    dataset: Optional[str] = field(default="ultrafeedback_binarized", metadata={"help": "the dataset used for training and evaluation "})

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

    generation_args: dict = field(
        default_factory=lambda: {
            "max_new_tokens": 356,
            "temperature": 0.2,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.9,
            "num_beams": 5,
            "min_length": 100,
            'num_return_sequences': 1,
            "sampling_strategy": "top_p_sampling"
        },
        metadata={"help": "arguments for model generation"}
    )
    
    temperature_list: List[float] = field(
        default_factory=lambda: [0.2],
        metadata={"help": "list of temperatures to evaluate models with"}
    )




def process_data_template(example):
    

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

def process_data_template_for_reward_model(row, gpt_friction = None):
    system_prompt_rm = (
        "Please rate the following friction intervention in light of the **dialogue history** of a *game* provided below. "
        "A friction intervention is a statement that acts as indirect persuasion and prompts participants to "
        "reevaluate their beliefs and assumptions about the task, primarily—but not exclusively—in response to new evidence "
        "that challenges their preconceived notions about the state of the task or the block weights."
    )

    friction_definition_game_definition_prompt_rm = (
        "*Game details and ground-truth*: The game is called 'Game of Weights.' The participants (P1, P2, and P3) are "
        "trying to determine the weight of various blocks. The blocks are of different colors and have specific weights in grams: "
        "the red block is 10 grams, the blue block is 10 grams, the green block is 20 grams, the purple block is 30 grams, and "
        "the yellow block is 50 grams. At the start of the game, participants are only allowed to weigh two blocks at a time, "
        "and they are told the weight of the red block. The participants must figure out how to determine the weight of each block. "
        "At the beginning of the game, they are unaware of the actual weights. Additionally, we inform the participants that they "
        "don’t need to use the scale's slider. The actual reason is that the blocks are in increments of 10 grams. "
        "The **dialogue history** is given below: "
    )

  
    prompt = (system_prompt_rm + row['context']).replace('\n', ' ')
    
    text = prompt + " " + f"</s> {row['agent_rationale']} {row['agent_friction']} </s>"
    if gpt_friction:
        text = prompt + " " + f"</s> {row['gpt_agent_friction']} </s>"
    
    return {
        'prompt': text,
       
    }



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


def normalize_block_name(block_text):
    """
    Normalize a block name to standardized format.
    
    Parameters:
    - block_text: String containing a block name
    
    Returns:
    - Normalized block name
    """
    if not block_text or not isinstance(block_text, str):
        return None
    
    # Remove parenthetical weight info and punctuation
    block_text = re.sub(r'\(.*?\)', '', block_text).strip()
    block_text = re.sub(r'[,.:;!?]', '', block_text).strip()
    
    # Check against standard block names (case-insensitive)
    for standard_block in STANDARD_BLOCKS:
        if standard_block.lower() in block_text.lower():
            return standard_block
    
    return None

def extract_resolved_blocks(resolved_blocks_list):
    """
    Extract and normalize block names from the resolved_blocks list.
    
    Parameters:
    - resolved_blocks_list: List of strings containing block names
    
    Returns:
    - Set of normalized block names
    """
    normalized_blocks = set()
    
    if not resolved_blocks_list:
        return normalized_blocks
    
    for block_text in resolved_blocks_list:
        normalized_block = normalize_block_name(block_text)
        if normalized_block:
            normalized_blocks.add(normalized_block)
    
    return normalized_blocks



def compute_dialogue_metrics(dialogue_data, agent_model, agent_tokenizer, reward_model, rm_tokenizer, generation_args):

    """
    Compute metrics for a single dialogue.
    
    Parameters:
    - dialogue_data: Dictionary containing data for a single dialogue
    
    Returns:
    - Dictionary with computed metrics for the dialogue
    """
    dialogue_metrics = {
        'dialogue_id': dialogue_data['dialog_id'],
        'total_turns': len(dialogue_data['turns']),
        'blocks_resolved_per_turn': [],
        'quality_metrics': [],
        'resolved_blocks_count': [],
        'turn_metrics': [],
        'personalities': dialogue_data.get('personalities', {}),
        'original_context': dialogue_data.get('original_context', ""),
        'gold_friction_bootstrap': dialogue_data.get('gold_friction_bootstrap', ""),
        'friction_scores': []
    }
    
    # Track resolved blocks - both GPT-reported and our independent detection
    resolved_blocks = set()
    prev_resolved_blocks = set()
    
    # For our independent detection
    detected_resolved_blocks = set()
    prev_detected_resolved_blocks = set()
    
#     print("len of dialogues", len(dialogue_data['turns']))
    
    # Process each turn
    for i, turn_data in enumerate(dialogue_data['turns']):
        # Add friction scores for each turn
        if 'parsed_gpt_response' in turn_data and 'friction_score' in turn_data['parsed_gpt_response']:
            dialogue_metrics['friction_scores'].append(turn_data['parsed_gpt_response']['friction_score'])
        
        # Compute turn metrics - this now uses our independent block detection
        turn_metrics = compute_turn_metrics(turn_data, i, resolved_blocks, 
                                prev_resolved_blocks, skip_bertscore=True,
                               agent_model = agent_model, agent_tokenizer = agent_tokenizer, reward_model= reward_model, rm_tokenizer = rm_tokenizer,
                                           generation_args = generation_args)
        dialogue_metrics['turn_metrics'].append(turn_metrics)
        
        # Update GPT-reported resolved blocks tracking (for comparison)
        if 'parsed_gpt_response' in turn_data and 'resolved_blocks' in turn_data['parsed_gpt_response']:
            curr_resolved = extract_resolved_blocks(turn_data['parsed_gpt_response']['resolved_blocks'])
            prev_resolved_blocks = resolved_blocks.copy()
            resolved_blocks = resolved_blocks.union(curr_resolved)
        
#         # Update our independently detected resolved blocks
#         if 'resolved_blocks' in turn_metrics:
#             prev_detected_resolved_blocks = detected_resolved_blocks.copy()
#             detected_resolved_blocks = set(turn_metrics['resolved_blocks'])
            
                # Update our independently detected resolved blocks
        if 'resolved_blocks' in turn_metrics:
            prev_detected_resolved_blocks = detected_resolved_blocks.copy()
            detected_resolved_blocks = detected_resolved_blocks.union(set(turn_metrics['resolved_blocks']))

        # Store current resolved blocks count and blocks resolved in this turn
        # (using our independent detection instead of GPT-reported)
        dialogue_metrics['resolved_blocks_count'].append(len(detected_resolved_blocks))
        blocks_in_turn = len(detected_resolved_blocks) - len(prev_detected_resolved_blocks)
        dialogue_metrics['blocks_resolved_per_turn'].append(blocks_in_turn)
    
    # Compute dialogue-level metrics using our independent detection
    dialogue_metrics['final_blocks_resolved'] = len(detected_resolved_blocks)
    dialogue_metrics['resolution_rate'] = len(detected_resolved_blocks) / len(STANDARD_BLOCKS)
    
    # List the actual resolved blocks
    dialogue_metrics['resolved_blocks'] = list(detected_resolved_blocks)
    
    # Store GPT-reported resolved blocks for comparison
    dialogue_metrics['gpt_reported_resolved_blocks'] = list(resolved_blocks)
    dialogue_metrics['gpt_reported_resolution_rate'] = len(resolved_blocks) / len(STANDARD_BLOCKS)
    
    # Check if all blocks were resolved
    all_blocks_resolved = len(detected_resolved_blocks) == len(STANDARD_BLOCKS)
    dialogue_metrics['all_blocks_resolved'] = all_blocks_resolved
    
    # Calculate turns until resolution if all blocks were resolved
    if all_blocks_resolved:
        for i, count in enumerate(dialogue_metrics['resolved_blocks_count']):
            if count == len(STANDARD_BLOCKS):
                dialogue_metrics['turns_until_resolution'] = i + 1
                break
    else:
        dialogue_metrics['turns_until_resolution'] = dialogue_metrics['total_turns']
    
    # Add agreement metrics between GPT-reported and independently detected blocks
    if len(resolved_blocks) > 0 or len(detected_resolved_blocks) > 0:
        overlap = len(resolved_blocks.intersection(detected_resolved_blocks))
        precision = overlap / len(resolved_blocks) if len(resolved_blocks) > 0 else 0
        recall = overlap / len(detected_resolved_blocks) if len(detected_resolved_blocks) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        dialogue_metrics['resolution_agreement'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'overlap_count': overlap
        }
    
    return dialogue_metrics

def compute_turn_metrics(turn_data, turn_index, resolved_blocks, prev_resolved_blocks, 
                         skip_bertscore=False, agent_model = None, 
                             agent_tokenizer = None, reward_model= None, rm_tokenizer = None, generation_args = None):
    """
    Compute metrics including counterfactual reward for a single turn.
    OBJECTIVE:
        First, uses the SFT context to compute its rewards. 
        Next, computes reward of any api based model in the dialog fields. 
        Finally, samples a generation for each trained friction agent baseline on the context (SFT), computes reward and logs!
    
    Parameters:
    - turn_data: Dictionary containing data for a single turn
    - turn_index: Index of the turn
    - resolved_blocks: Set of blocks resolved so far
    - prev_resolved_blocks: Set of blocks resolved before this turn
    - skip_bertscore: Skip computing BERTScore metrics (default: False)
    
    Returns:
    - Dictionary with computed metrics for the turn
    """
    turn_metrics = {
        'turn_number': turn_index + 1,
        'quality': {},
        'persuasiveness': {},
        'timestamp': turn_data.get('timestamp', ""),
        'parsed_friction': turn_data.get('parsed_friction', ""),
        'gpt_friction': turn_data.get('parsed_gpt_response', {}).get('friction_statement', ""),
        'turn_level_friction_score': turn_data.get('parsed_gpt_response', {}).get('friction_score', "")
    }
    device = 'cuda:1'
    # Quality metrics - Semantic similarity between friction agent and GPT friction
    if 'parsed_friction' in turn_data and 'parsed_gpt_response' in turn_data and 'friction_statement' in turn_data['parsed_gpt_response'] and turn_data['parsed_gpt_response']['friction_statement']:
        agent_friction = turn_data['parsed_friction']
       
        #get agent rationale
        agent_rationale = turn_data['rationale']
        dialogue_before_friction = turn_data['dialogue_before_friction'] # this is the dialogue history
        ref_policy_inputs = {}
        
        ref_policy_inputs['context'] = dialogue_before_friction
        ref_policy_inputs['agent_friction'] = agent_friction
        ref_policy_inputs['agent_rationale'] = agent_rationale
        
        
        if len(dialogue_before_friction.split()) > 6000:
            original_length = len(agent_tokenizer.encode(dialogue_before_friction))
            print(f"[DEBUG] Original dialogue history: {original_length} tokens, {len(dialogue_before_friction.split())} words")
            ref_policy_inputs['context'] = truncate_dialogue_history(
                ref_policy_inputs['context'], 
                agent_tokenizer, 
                max_tokens=6000
            )
            truncated_length = len(agent_tokenizer.encode(ref_policy_inputs['context']))
            print(f"[DEBUG] After truncation: {truncated_length} tokens ({(truncated_length/original_length)*100:.1f}% of original)")
        
        
        # process in the input prompt for the RM
        ref_policy_inputs.update(process_data_template_for_reward_model(ref_policy_inputs))
        gpt_friction = turn_data['parsed_gpt_response']['friction_statement']
        # For the GPT agent friction
        gpt_policy_inputs = ref_policy_inputs.copy()  # Create a copy 
        
        gpt_policy_inputs['gpt_agent_friction'] = gpt_friction  # Add the GPT friction
        gpt_policy_inputs.update(process_data_template_for_reward_model(gpt_policy_inputs, gpt_friction=True))


        # Wrap reward model operations in torch.no_grad()
        with torch.no_grad():
            prompt_list = [ref_policy_inputs['prompt'], gpt_policy_inputs['prompt']]
            tokenized_inputs = rm_tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            
            # Run the model on the batch
            reward_outputs = reward_model(**tokenized_inputs)
            
            # Extract both scores
            batch_scores = reward_outputs.logits.squeeze(-1).cpu()
                # Store SFT and GPT scores
            sft_score = batch_scores[0].item() if batch_scores.numel() > 0 else 0
            gpt_score = batch_scores[1].item() if batch_scores.numel() > 1 else 0
            turn_metrics['reward_sft'] = sft_score
            turn_metrics['reward_gpt'] = gpt_score
            turn_metrics['reward_margin_gpt_sft'] = gpt_score - sft_score
            turn_metrics['gpt_win_vs_sft'] = gpt_score > sft_score
            
            print("reward_sft:", sft_score)
            print("reward_gpt:", gpt_score)
            print("margin (GPT - SFT):", gpt_score - sft_score)
            
            # Clean up memory
            del tokenized_inputs, reward_outputs, batch_scores
            torch.cuda.empty_cache()
            gc.collect()

 
            
            ref_policy_inputs.update(process_data_template(ref_policy_inputs))
            # print("ref_policy_inputs", ref_policy_inputs['prompt'])
            
            # generate responsed from all other model baselines
            generated_texts, all_generated_texts = generate_multiple_sequences_with_intrinsic_metrics(
                                    agent_model, 
                                    agent_tokenizer, 
                                    ref_policy_inputs['prompt'], 
                                    generation_args, 
                                    None,
                                    strategy="top_p_sampling", 
                                    batched=True, 
                                )

            
            
            
            # Process the generated text
            if generated_texts and isinstance(generated_texts, list):
                text_to_parse = generated_texts[0][0] if (generated_texts[0] and isinstance(generated_texts[0], list)) else generated_texts[0]
                turn_data['model_generated_text'] = text_to_parse
                
            tags_for_parsing = ["friction", "rationale", "t", "b"]
            parsed_frictive_states_and_friction = parse_tags_robust(text_to_parse, tags_for_parsing)

            # Extract components
            friction_intervention = ' '.join(parsed_frictive_states_and_friction.get('friction', []))
            if not friction_intervention:
                friction_intervention = handle_friction_logic(text_to_parse)

            task_state = ' '.join(parsed_frictive_states_and_friction.get('t', []))
            belief_state = ' '.join(parsed_frictive_states_and_friction.get('b', []))
            rationale = ' '.join(parsed_frictive_states_and_friction.get('rationale', []))
            


            if len(ref_policy_inputs['context'] .split()) > 6000:
                original_length = len(agent_tokenizer.encode(ref_policy_inputs['context']))
                print(f"[DEBUG] Original dialogue history: {original_length} tokens, {len(ref_policy_inputs['context'] .split())} words")
                ref_policy_inputs['context'] = truncate_dialogue_history(
                    ref_policy_inputs['context'], 
                    agent_tokenizer, 
                    max_tokens=6000
                )
                truncated_length = len(agent_tokenizer.encode(ref_policy_inputs['context']))
                print(f"[DEBUG] After truncation: {truncated_length} tokens ({(truncated_length/original_length)*100:.1f}% of original)")
        
            agent_policy_inputs = {
                'agent_friction': friction_intervention,
                'agent_rationale': rationale,
                'context': ref_policy_inputs['context']  # Keep the same context
            }
            agent_policy_inputs.update(process_data_template_for_reward_model(agent_policy_inputs))
            # Same for the second reward model call
            tokenized_inputs = rm_tokenizer(agent_policy_inputs['prompt'], return_tensors="pt", padding=True, truncation=True, max_length=1024)

            reward_outputs = reward_model(**tokenized_inputs)
            agent_score = reward_outputs.logits.squeeze(-1).cpu().item() if reward_outputs.logits.numel() == 1 else reward_outputs.logits.squeeze(-1).cpu().tolist()

            # Store agent score and comparisons
            turn_metrics['reward_agent'] = agent_score
            turn_metrics['reward_margin_agent_sft'] = agent_score - sft_score
            turn_metrics['reward_margin_gpt_agent'] = gpt_score - agent_score
            turn_metrics['agent_win_vs_sft'] = agent_score > sft_score
            turn_metrics['gpt_win_vs_agent'] = gpt_score > agent_score
            print("reward_agent", agent_score)
            print("margin (reward_agent - SFT):", agent_score - sft_score)
            print("margin (reward_agent - GPT):", agent_score - gpt_score)
            del tokenized_inputs, reward_outputs, agent_score
            torch.cuda.empty_cache()
            gc.collect()


            turn_metrics['agent_friction'] = agent_policy_inputs.get('agent_friction', '')
            turn_metrics['agent_rationale'] = agent_policy_inputs.get('agent_rationale', '')
            # # Or just save prompt length
            turn_metrics['agent_prompt_length'] = len(agent_policy_inputs.get('context', ''))


    return turn_metrics

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

def subset_model_data(all_models_data, max_dialogues=2, max_turns=2):
    """
    Create a subset of model data with limited dialogues and turns.
    
    Args:
        all_models_data (dict): Dictionary containing all models' data
        max_dialogues (int): Maximum number of dialogues to keep per model
        max_turns (int): Maximum number of turns to keep per dialogue
        
    Returns:
        dict: Subset of the original data with limited dialogues and turns
    """
    all_models_data_subset = {}
    
    # Debug information
    print(f"[INFO] Subsetting data: keeping {max_dialogues} dialogues with {max_turns} turns each")
    
    for model_name, model_data in all_models_data.items():
        # Take only the specified number of dialogues for each model
        model_dialogues_subset = list(model_data.values())[:max_dialogues]
        
        # Trim each dialogue to have only the specified number of turns
        for dialogue in model_dialogues_subset:
            if 'turns' in dialogue and len(dialogue['turns']) > max_turns:
                dialogue['turns'] = dialogue['turns'][:max_turns]
        
        # Add to the new subset
        all_models_data_subset[model_name] = {
            list(model_data.keys())[i]: model_dialogues_subset[i] 
            for i in range(min(max_dialogues, len(model_data)))
        }
    
    # Print summary of the resulting subset
    total_dialogues = sum(len(model_data) for model_data in all_models_data_subset.values())
    print(f"[INFO] Created subset with {len(all_models_data_subset)} models, {total_dialogues} total dialogues")
    
    return all_models_data_subset


def calculate_statistical_significance(all_results, tests=["ttest", "mannwhitney", "bootstrap"]):
    """
    Perform multiple statistical tests to compare model performances.
    
    Args:
        all_results (dict): Dictionary with model metrics where the middle field (e.g., temperature) exists
        tests (list): Statistical tests to perform
        
    Returns:
        dict: Dictionary with statistical test results
    """
    import scipy.stats as stats
    import numpy as np
    import pandas as pd
    
    # Extract model names
    model_names = list(all_results.keys())
    if len(model_names) < 2:
        print("Need at least 2 models to perform statistical tests")
        return {}

    # Initialize results dictionary
    test_results = {
        'pairwise_tests': [],
        'summary': {}
    }
    
    # Define metrics to test
    metrics_to_test = [
        'avg_agent_reward', 
        'agent_win_rate_vs_sft',
        'avg_gpt_sft_margin'
    ]
    
    # Collect raw data across models and temperature settings
    raw_data = {}
    
    for model_name in model_names:
        raw_data[model_name] = {}  # Ensure model-level key exists
        
        for temp_setting, model_data in all_results[model_name].items():  # Iterate over temperature settings
            raw_data[model_name][temp_setting] = {}  # Store per temperature
            
            for metric in metrics_to_test:
                values = []
                for dialogue in model_data['dialogues']:
                    for turn in dialogue['turn_metrics']:
                        if metric == 'avg_agent_reward' and 'reward_agent' in turn:
                            values.append(turn['reward_agent'])
                        elif metric == 'agent_win_rate_vs_sft' and 'agent_win_vs_sft' in turn:
                            values.append(1 if turn['agent_win_vs_sft'] else 0)
                        elif metric == 'avg_gpt_sft_margin' and 'reward_margin_gpt_sft' in turn:
                            values.append(turn['reward_margin_gpt_sft'])
                
                raw_data[model_name][temp_setting][metric] = values
    
    # Perform pairwise tests for each temperature setting within models
    for model1 in model_names:
        for model2 in model_names:
            if model1 == model2:
                continue  # Skip self-comparisons
            
            for temp1 in all_results[model1]:
                for temp2 in all_results[model2]:
                    pair_results = {
                        'model1': model1,
                        'temperature1': temp1,
                        'model2': model2,
                        'temperature2': temp2,
                        'tests': {}
                    }
                    
                    for metric_key in metrics_to_test:
                        data1 = raw_data[model1][temp1].get(metric_key, [])
                        data2 = raw_data[model2][temp2].get(metric_key, [])
                        
                        if len(data1) > 0 and len(data2) > 0:
                            mean1, mean2 = np.mean(data1), np.mean(data2)
                            std1, std2 = np.std(data1), np.std(data2)

                            metric_results = {
                                'mean1': float(mean1),
                                'mean2': float(mean2),
                                'std1': float(std1),
                                'std2': float(std2),
                                'test_results': {}
                            }

                            # Effect size: Cohen's d
                            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                            cohens_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
                            metric_results['cohens_d'] = float(cohens_d)

                            # Statistical tests
                            if "ttest" in tests:
                                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                                metric_results['test_results']['ttest'] = {
                                    't_stat': float(t_stat),
                                    'p_value': float(p_value),
                                    'significant': p_value < 0.05,
                                    'better_model': model1 if mean1 > mean2 else model2
                                }

                            if "mannwhitney" in tests:
                                u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                                metric_results['test_results']['mannwhitney'] = {
                                    'u_stat': float(u_stat),
                                    'p_value': float(p_value),
                                    'significant': p_value < 0.05,
                                    'better_model': model1 if mean1 > mean2 else model2
                                }

                            if "bootstrap" in tests:
                                n_bootstrap = 1000
                                observed_diff = np.mean(data1) - np.mean(data2)
                                combined = np.concatenate([data1, data2])
                                count_exceeds = sum(
                                    abs(np.mean(np.random.choice(combined, size=len(data1))) - np.mean(np.random.choice(combined, size=len(data2))))
                                    >= abs(observed_diff)
                                    for _ in range(n_bootstrap)
                                )
                                p_value = count_exceeds / n_bootstrap
                                metric_results['test_results']['bootstrap'] = {
                                    'observed_diff': float(observed_diff),
                                    'p_value': float(p_value),
                                    'significant': p_value < 0.05,
                                    'better_model': model1 if mean1 > mean2 else model2
                                }

                            pair_results['tests'][metric_key] = metric_results

                    test_results['pairwise_tests'].append(pair_results)

    return test_results


def main_multi_model(results_log_file, models_list, script_args, output_folder = 'COUNTERFACTUAL_metrics_output'):
    """
    Main function for computing and reporting metrics across multiple models.
    This function loads the combined JSON file containing results for multiple models,
    computes metrics for each model, and saves the results.
    """
  
    data_file = f"{results_log_file}"
    output_dir = "friction_role_play_evaluation_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_folder, exist_ok=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    all_models_data = results_log_file
 
    # Dictionary to store metrics for all models
    all_models_metrics = {}
    
    # Dictionary for comparative analysis across models
    comparative_metrics = {
    'quality': {},
    'efficiency': {},
    'persuasiveness': {},
    'friction_scores': {}   
}
    
    generation_args = {
    "max_new_tokens": 356,
    "temperature": 0.1,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.9,
    "num_beams": 5,
    "min_length": 100,
    'num_return_sequences': 1,
    "sampling_strategy": "top_p_sampling"
    }
    
    #LIST OF ALL FRICTION AGENT MODELS
    
    reward_model_name = "friction_rm_alldata_results"
    device = 'cuda:1'
    torch_dtype = torch.bfloat16
 
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        num_labels=1,  # Set to 1 for reward model classification
        trust_remote_code=True,
         device_map="auto",
        torch_dtype=torch.float16,

        )

    rm_tokenizer = AutoTokenizer.from_pretrained(
    reward_model_name, 
    trust_remote_code=True, 
    use_fast=True
    )
    all_results = {}
    for agent_model_name in tqdm(models_list, desc="Processing Models"):
        loading_model_name = agent_model_name

        if "/" in agent_model_name:
            parts = agent_model_name.split("/")
            if len(parts) >= 2:
                # Combine the first two parts
                agent_model_name = parts[0] + "_" + parts[1]
      
        print(f"\n===== Processing Model: {agent_model_name} =====\n")
    
 

        lora_model = AutoModelForCausalLM.from_pretrained(
            script_args.base_model_name_or_path,
            device_map="auto",

            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        trust_remote_code=True,

        )

        lora_model = PeftModel.from_pretrained(
        lora_model,
        loading_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
        # Merge the model
        print("Merging LoRA adapter...")
        merged_model = lora_model.merge_and_unload()


        tokenizer = AutoTokenizer.from_pretrained(loading_model_name)
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.padding_side = "right"

 
 
        for model_name, model_data in tqdm(all_models_data.items(), desc=f"Processing {agent_model_name}", leave=False):
            for temperature in script_args.temperature_list:
                if model_name == "SFT": #hack but works! 
                    current_generation_args = script_args.generation_args.copy()
                    current_generation_args["temperature"] = temperature
                # if 'sft' in model_name:
                    print(f"\n\n===== PROCESSING AGENT, REF MODEL, TEMP: {agent_model_name}{model_name} {temperature} =====\n")
                    print(f"\n\n===== PROCESSING GENERATION ARGS: {current_generation_args['temperature']} =====\n")

                    # this loop will now essentially run for the SFT model only 

                    metrics = compute_metrics(model_data, merged_model,tokenizer, reward_model, rm_tokenizer, current_generation_args)
                    # Initialize variables for aggregate calculations
                                
                    agent_win_vs_sft_count = 0
                    gpt_win_vs_sft_count = 0
                    gpt_win_vs_agent_count = 0
                    total_turns = 0

                    # Arrays for individual values
                    agent_rewards = []
                    sft_rewards = []
                    gpt_rewards = []
                    agent_sft_margins = []
                    gpt_sft_margins = []
                    gpt_agent_margins = []
                    agent_wins_vs_sft = []
                    gpt_wins_vs_sft = []
                    gpt_wins_vs_agent = []

                        # Process each dialogue's turns
                                    # Process each dialogue's turns
                    for dialogue in metrics['dialogues']:
                        for turn in dialogue['turn_metrics']:
                            if 'reward_agent' in turn and 'reward_sft' in turn and 'reward_gpt' in turn:
                                agent_reward = turn['reward_agent']
                                sft_reward = turn['reward_sft']
                                gpt_reward = turn['reward_gpt']
                                
                                # Track counts
                                total_turns += 1
                                
                                # Store individual values for std calculation
                                agent_rewards.append(agent_reward)
                                sft_rewards.append(sft_reward)
                                gpt_rewards.append(gpt_reward)
                                
                                # Margins
                                agent_sft_margin = turn['reward_margin_agent_sft']
                                gpt_sft_margin = turn['reward_margin_gpt_sft']
                                gpt_agent_margin = turn['reward_margin_gpt_agent']
                                agent_sft_margins.append(agent_sft_margin)
                                gpt_sft_margins.append(gpt_sft_margin)
                                gpt_agent_margins.append(gpt_agent_margin)
                                
                                # Wins
                                agent_win_vs_sft = turn['agent_win_vs_sft']
                                gpt_win_vs_sft = turn['gpt_win_vs_sft']
                                gpt_win_vs_agent = turn['gpt_win_vs_agent']
                                
                                agent_wins_vs_sft.append(1 if agent_win_vs_sft else 0)
                                gpt_wins_vs_sft.append(1 if gpt_win_vs_sft else 0)
                                gpt_wins_vs_agent.append(1 if gpt_win_vs_agent else 0)
                                
                                agent_win_vs_sft_count += 1 if agent_win_vs_sft else 0
                                gpt_win_vs_sft_count += 1 if gpt_win_vs_sft else 0
                                gpt_win_vs_agent_count += 1 if gpt_win_vs_agent else 0

                    # Calculate standard deviations
                
                    agent_reward_std = np.std(agent_rewards) if agent_rewards else 0
                    sft_reward_std = np.std(sft_rewards) if sft_rewards else 0
                    gpt_reward_std = np.std(gpt_rewards) if gpt_rewards else 0

                    agent_sft_margin_std = np.std(agent_sft_margins) if agent_sft_margins else 0
                    gpt_sft_margin_std = np.std(gpt_sft_margins) if gpt_sft_margins else 0
                    gpt_agent_margin_std = np.std(gpt_agent_margins) if gpt_agent_margins else 0

                    agent_win_rate_std = np.std(agent_wins_vs_sft) if agent_wins_vs_sft else 0
                    gpt_sft_win_rate_std = np.std(gpt_wins_vs_sft) if gpt_wins_vs_sft else 0
                    gpt_agent_win_rate_std = np.std(gpt_wins_vs_agent) if gpt_wins_vs_agent else 0

                    # Add aggregate metrics
                    metrics['aggregates'] = {
                        # Raw scores
                        'avg_agent_reward': np.mean(agent_rewards) if agent_rewards else 0,
                        'std_agent_reward': agent_reward_std,
                        'avg_sft_reward': np.mean(sft_rewards) if sft_rewards else 0,
                        'std_sft_reward': sft_reward_std,
                        'avg_gpt_reward': np.mean(gpt_rewards) if gpt_rewards else 0,
                        'std_gpt_reward': gpt_reward_std,
                        
                        # Margins
                        'avg_agent_sft_margin': np.mean(agent_sft_margins) if agent_sft_margins else 0,
                        'std_agent_sft_margin': agent_sft_margin_std,
                        'avg_gpt_sft_margin': np.mean(gpt_sft_margins) if gpt_sft_margins else 0,
                        'std_gpt_sft_margin': gpt_sft_margin_std,
                        'avg_gpt_agent_margin': np.mean(gpt_agent_margins) if gpt_agent_margins else 0,
                        'std_gpt_agent_margin': gpt_agent_margin_std,
                        
                        # Win rates
                        'agent_win_rate_vs_sft': agent_win_vs_sft_count / total_turns if total_turns > 0 else 0,
                        'std_agent_win_rate_vs_sft': agent_win_rate_std,
                        'gpt_win_rate_vs_sft': gpt_win_vs_sft_count / total_turns if total_turns > 0 else 0,
                        'std_gpt_win_rate_vs_sft': gpt_sft_win_rate_std,
                        'gpt_win_rate_vs_agent': gpt_win_vs_agent_count / total_turns if total_turns > 0 else 0,
                        'std_gpt_win_rate_vs_agent': gpt_agent_win_rate_std,
                        
                        # Counts
                        'total_turns': total_turns,
                        'agent_win_vs_sft_count': agent_win_vs_sft_count,
                        'gpt_win_vs_sft_count': gpt_win_vs_sft_count,
                        'gpt_win_vs_agent_count': gpt_win_vs_agent_count
                    }

                    # Store results using the new middle key
                    if agent_model_name not in all_results:
                        all_results[agent_model_name] = {}  # Ensure the outer key exists
                
                    # Store in master dictionary using the agent model name as key
                    all_results[agent_model_name][temperature]= metrics
                
  
    # Save results
    # Create output paths with timestamp
    agg_csv_path = os.path.join(output_folder, f"aggregate_metrics_by_model_{timestamp}.csv")
    pkl_path = os.path.join(output_folder, f"full_reward_metrics_roleplay_COUNTERFACTUAL_{timestamp}.pkl")
    stats_path =  os.path.join(output_folder, f"full_reward_STATS_SIGNIFICANCE_metrics_roleplay_COUNTERFACTUAL_{timestamp}.pkl")
    

    # Create dataframe for aggregate metrics across all models
    aggregate_rows = []
    for agent_model_name, model_data in all_results.items():
        for temperature, metrics in model_data.items():  # Iterate over middle fields
            if 'aggregates' in metrics:
                row = {'model': agent_model_name, 'temperature': temperature}  # Include middle field
                row.update(metrics['aggregates'])
                aggregate_rows.append(row)

    # Create DataFrame and Save
    agg_df = pd.DataFrame(aggregate_rows)
    agg_df.to_csv(agg_csv_path, index=False)
    print(f"Saved aggregate metrics to: {agg_csv_path}")

    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Saved full metrics to: {pkl_path}")

    # get statistical tests and p values
    statistical_results = calculate_statistical_significance(all_results)
    with open(stats_path, 'wb') as f:
        pickle.dump(statistical_results, f)
    print(f"Saved full metrics to: {stats_path}")

    return metrics


if __name__ == "__main__":
    # Define arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args, _ = parser.parse_known_args()

    # Set seed for reproducibility
    set_seed(script_args.seed)

    #load model and tokenizer

     # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
        
    models_list = ['SFT', 'DPO', 'PPO', 'friction_plus'
  
    ]
    
    json_data_1 = "logged_data_models.json"

    with open(json_data_1, 'r') as f:
        all_models_data = json.load(f)
    
    print("LOGGED DATA MODELS", all_models_data.keys())

    # RUN MAIN LOOP OF COUNTERFACTUAL REWARD MODEL EVALUATION--> passes through all models, 
    #then all dialog/stored context of SFT model, then turns --> computes sampled counterfactual rewards of baselines
    final_metrics = main_multi_model(all_models_data, models_list, script_args)