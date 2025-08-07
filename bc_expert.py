import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pickle
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from transformers import DataCollatorForLanguageModeling
from itertools import combinations
from torch.utils.data import DataLoader
import torch
import warnings
from accelerate impo
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
# from trl.import_utils import is_npu_available, is_xpu_available
from trl.trainer import ConstantLengthDataset
import bitsandbytes as bnb
# optim_8bit = bnb.optim.Adam8bit

@dataclass
class ScriptArguments:
 
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={"help": "the model name"})
 
    dataset_name: Optional[str] = field(default="wtd_simulated", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    use_bnb: Optional[bool] = field(default=True, metadata={"help": "whether to use BitsAndBytes"})

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser((ScriptArguments, SFTConfig))
script_args, training_args = parser.parse_args_into_dataclasses()
# training_args.optim = optim_8bit
print("training optimizer sanity check",training_args.optim)
# print("Full Training args",training_args)
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

if training_args.group_by_length and training_args.packing:
    raise ValueError("Cannot use both packing and group by length")

# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
if training_args.gradient_checkpointing:
    raise ValueError("gradient_checkpointing not supported")

set_seed(training_args.seed)

def create_preference_dataset_friction(data):
    """
    Creates the simplest preference dataset where original friction is chosen 
    and lowest scored relevant statement is rejected
    """
    pairs = []
    
    for key, entry in data.items():
        original_friction = entry['friction_data_original']['friction_statement']
        original_rationale = entry['friction_data_original']['rationale']
        relevant_statements = entry['gpt_friction_rogue_rankings']['relevant_statements']
        
        # Find the lowest scored relevant statement
        lowest_relevant = min(relevant_statements, key=lambda x: x['relevance_score'])
        
        pair = {
            'chosen': original_friction,
            'rejected': lowest_relevant['statement'],
            'chosen_score': 11,  # Assigning higher score to original
            'rejected_score': lowest_relevant['relevance_score'],
            'context': entry['friction_data_original']['previous_utterance_history'],
            'task_state': entry['friction_data_original']['task_summary'],
            'belief_state': entry['friction_data_original']['belief_summary'],
           'dialog_id': entry['friction_data_original']['dialog_id'],
            'friction_participant': entry['friction_data_original']['friction_participant'],
          'chosen_rationale': original_rationale,
            'rejected_rationale': lowest_relevant['rationale'],
            'rationale_present': entry['friction_data_original']['rationale_present']
        
        }
        pairs.append(pair)
    
    return Dataset.from_pandas(pd.DataFrame(pairs))

def create_train_test_split(dataset, test_size=0.1, seed=42):
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({
        'train': split['train'],
        'test': split['test']
    })


def chars_token_ratio(data, tokenizer):
    """
    Compute the average number of characters per token in the dataset.
    
    Args:
        data: Dataset with messages field
        tokenizer: Tokenizer to use
        
    Returns:
        Average number of characters per token
    """
    total_characters = 0
    total_tokens = 0
    
    # Process a subset of the data for efficiency if the dataset is large
    sample_size = min(len(data), 400)
    
    for i in range(sample_size):
        example = data[i]
        # Extract messages
        if 'messages' in example:
            # Concatenate all message contents
            chat_text = ""
            for message in example['messages']:
                if 'content' in message and isinstance(message['content'], str):
                    chat_text += message['content'] + " "
            
            # Count characters and tokens
            if chat_text:
                total_characters += len(chat_text)
                total_tokens += len(tokenizer.encode(chat_text))
    
    if total_tokens == 0:
        return 5.0  # Default fallback value
        
    return total_characters / total_tokens

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text

def prepare_friction_prompt_response(example):
    """Prepare the friction generation prompt and response for SFT training."""
    #old friction prompt
    # system_prompt_rm = (
    # "You are an expert in collaborative task analysis and personality-driven communication. "
    # "Your task is to generate nuanced friction statements within a dialogue. "
    # "Given the **dialogue history** involving three participants and the *game details*, "
    # "generate a <friction> statement that acts as indirect persuasion. This statement should "
    # "encourage the participants to reevaluate their beliefs and assumptions about the task. "
    # "Additionally, provide a <rationale> or explanation for your friction statement. Base your reasoning "
    # "on evidence from the dialogue, focusing on elements such as: "
    # "- Incorrect assumptions "
    # "- False beliefs "
    # "- Rash decisions "
    # "- Missing evidence ")

    # more intuitive friction generation system prompt: first predicts the task, then beliefs, then explian why friction is needed before generating the friction intervention
    system_prompt_rm = (
    "You are an expert in collaborative task analysis and personality-driven communication. Think step by step. "
    "Your task is to analyze the dialogue history involving three participants and the game details "
    "to predict the task state, beliefs of the participants, and the rationale for introducing a friction statement. "
    "Finally, generate a nuanced friction statement based on your analysis.\n\n"
    "1. Predict the task-related context and enclose it between the markers `<t>` and `</t>`.\n\n"
    "2. Predict the belief-related context for the participants and enclose it between the markers `<b>` and `</b>`.\n\n"
    "3. Provide a rationale for why a friction statement is needed. This rationale must be enclosed between the "
    "markers `<rationale>` and `</rationale>`. Base your reasoning on evidence from the dialogue, focusing on elements such as:\n"
    "- Incorrect assumptions\n"
    "- False beliefs\n"
    "- Rash decisions\n"
    "- Missing evidence.\n\n"
    "4. Generate the friction statement, ensuring it is enclosed between the markers `<friction>` and `</friction>`. "
    "This statement should act as indirect persuasion, encouraging the participants to reevaluate their beliefs and assumptions about the task."
)


    # friction_definition_game_definition_prompt_rm = (
    #     "*Game details and ground-truth*: The game is called 'Game of Weights.' The participants (P1, P2, and P3) are "
    #     "trying to determine the weight of various blocks. The blocks are of different colors and have specific weights in grams: "
    #     "the red block is 10 grams, the blue block is 10 grams, the green block is 20 grams, the purple block is 30 grams, and "
    #     "the yellow block is 50 grams. At the start of the game, participants are only allowed to weigh two blocks at a time, "
    #     "and they are told the weight of the red block. The participants must figure out how to determine the weight of each block. "
    #     "At the beginning of the game, they are unaware of the actual weights. Additionally, we inform the participants that they "
    #     "don’t need to use the scale's slider. The actual reason is that the blocks are in increments of 10 grams. "
    #     "The **dialogue history** is given below: "
    # )

    friction_definition_game_definition_prompt_rm = (
    "The game is called 'Game of Weights,' where participants (P1, P2, and P3) determine the weights of colored blocks. "
    "Participants can weigh two blocks at a time and know the weight of the red block. "
    "They must deduce the weights of other blocks. "
    "The dialogue history is provided below:"
)

    # "Be specific and ensure that your response clearly addresses the dynamics in the dialogue.")

    # text = f"Question: {system_prompt_rm} {friction_definition_game_definition_prompt_rm}. {example['context']}\n\nAnswer: <friction> {example['chosen']}. <rationale>: {example['chosen_rationale']}" # old sft prompt format
    # the below prompt is formatted acc to llama3 instruction format from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/14
    # this excludes the game definition prompt since
    text = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    f"{system_prompt_rm}. {friction_definition_game_definition_prompt_rm}\n\n"
    f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    f"{example['context']}\n\n"
    f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    f"### Assistant: <t> {example['task_state']} </t>\n"
    f"        <b> {example['belief_state']} </b>\n"
    f"        <rationale>: {example['rationale']} </rationale>\n"
    f"        <friction> {example['friction_statement']} </friction>\n"
    f"<|eot_id|>"
)

    return text



class DataCollatorForFullAssistantLM(DataCollatorForCompletionOnlyLM):
    """
    Modified version of DataCollatorForCompletionOnlyLM that includes all of the assistant's tokens
    in the loss computation, including the template tokens.
    """
    
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
#         batch = super(DataCollatorForLanguageModeling, self).torch_call(examples)
#         batch = super().torch_call(examples)
        # print("DEBUG EXAMPLE TYPES:", type(examples[0]), examples[0])
        batch =DataCollatorForLanguageModeling.torch_call(self, examples)
 

        for i in range(len(examples)):
            response_token_ids_idxs = []
            human_token_ids_idxs = []
            
            # Find all assistant response starts
            for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                if (self.response_token_ids == 
                    batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()):
                    # Include the template tokens by not adding their length
                    response_token_ids_idxs.append(assistant_idx)
            
            if len(response_token_ids_idxs) == 0:
                warnings.warn(
                    f"Could not find response key `{self.response_template}` in instance {i}. "
                    "This instance will be ignored in loss calculation.",
                    UserWarning,
                )
                batch["labels"][i, :] = self.ignore_index
                continue
            
            # Find all human instruction starts if template is provided
            if self.instruction_template is not None:
                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)
                
                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in instance {i}. "
                        "This instance will be ignored in loss calculation.",
                        UserWarning,
                    )
                    batch["labels"][i, :] = self.ignore_index
                    continue
                
                # Handle case where first human token comes after first assistant token
                if human_token_ids_idxs[0] > response_token_ids_idxs[0]:
                    human_token_ids_idxs = [0] + human_token_ids_idxs
            else:
                # If no instruction template, treat everything before first response as human
                human_token_ids_idxs = [0]
            
            # Set all non-assistant tokens to ignore_index
            current_idx = 0
            for j in range(len(response_token_ids_idxs)):
                # Get start of assistant response
                assistant_start = response_token_ids_idxs[j]
                
                # Get end of assistant response (start of next human or end of sequence)
                if j + 1 < len(human_token_ids_idxs):
                    assistant_end = human_token_ids_idxs[j + 1]
                else:
                    assistant_end = len(batch["labels"][i])
                
                # Make pytorch loss function ignore all non-assistant tokens
                if current_idx < assistant_start:
                    batch["labels"][i, current_idx:assistant_start] = self.ignore_index
                
                # Keep assistant tokens for loss computation (don't mask them)
                current_idx = assistant_end
            
            # Mask any remaining tokens after the last assistant segment
            if current_idx < len(batch["labels"][i]):
                batch["labels"][i, current_idx:] = self.ignore_index
                
        return batch

def create_chat_from_turns(dialogue_data, max_turns=None, include_belief_state=False):
    """
    Create a chat list from dialogue turns data.
    Adds simple system prompt for BC expert friction training
    
    Args:
        dialogue_data: Dictionary containing the dialogue data with 'turns' field
        max_turns: Maximum number of turns to include (None for all)
        include_belief_state: Whether to include belief_state field in assistant messages
        
    Returns:
        List of chat messages with alternating user/assistant roles
    """
    if 'turns' not in dialogue_data or not dialogue_data['turns']:
        return []
    
    turns = dialogue_data['turns']
    if max_turns is not None:
        turns = turns[:min(max_turns, len(turns))]
    
    chat = []
    
    # first append the system role at the top of each conversation
    system_prompt  = "You are an expert in collaborative task analysis and reasoning. You assist a group solving the Wason card selection task by prompting them to consider missing perspectives. The rule to test: 'All cards with vowels have even numbers on the other side. Generate a friction intervention that fosters self-reflection, realigns understanding, and supports collaboration."
    
    
    # First user message from dialogue_before_friction in first turn
    if turns and 'dialogue_before_friction' in turns[0]:
        
        chat.append({
            "role": "system",
            "content": system_prompt
        })
      
        chat.append({
            "role": "user",
            "content": turns[0]['dialogue_before_friction']
        })
    
    # Process each turn
    total_turns = len(turns)
    for index, turn in enumerate(turns):
        # Assistant message from parsed_friction
        assistant_content = ""
        
        if include_belief_state and 'belief_state' in turn:
            assistant_content = f"Belief State: {turn['belief_state']}\n\n"
            
        if 'parsed_friction' in turn:
            assistant_content += turn['parsed_friction']
            
        if assistant_content:
            chat.append({
                "role": "assistant",
                "content": assistant_content.strip()
            })
        
        # User message from gpt_utterances (for turns after the first)
        
        if turn['turn_number'] > 0 and 'gpt_utterances' in turn:
            if index != total_turns -1:
                user_content = ""

                # Join all utterances with newlines
                if isinstance(turn['gpt_utterances'], list):
                    user_content = "\n".join(turn['gpt_utterances'])
                else:
                    user_content = str(turn['gpt_utterances'])

                if user_content:
                    chat.append({
                        "role": "user",
                        "content": user_content.strip()
                    })
            else:
                continue
    
    return chat

# Usage example
def process_all_dialogues(deli_all_models_data, max_turns=None, include_belief_state=False):
    all_chats = {}
    
    for model_name, model_data in deli_all_models_data.items():
        all_chats[model_name] = {}
        
        for dialogue_id, dialogue in model_data.items():
            chat = create_chat_from_turns(
                dialogue, 
                max_turns=max_turns,
                include_belief_state=include_belief_state
            )
            all_chats[model_name][dialogue_id] = chat
    
    return all_chats
 

def flatten_chats_only(all_chats):
    """
    Flatten the nested chats structure into a simple list of chat sequences.
    
    Args:
        all_chats: Nested dictionary with model_name -> dialogue_id -> chat structure
        
    Returns:
        List of chat lists, where each chat list contains user/assistant messages
    """
    flattened_chats = []
    
    for model_name, model_dialogues in all_chats.items():
        for dialogue_id, chat in model_dialogues.items():
            # Only add the chat list itself, not the metadata
            flattened_chats.append(chat)
    
    return flattened_chats



def format_chat_template(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

def generate_multiturn_dataset(tokenizer, args, train_data, valid_data, seed=None):
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    # REMOVE THE EARLY RETURN
    # return train_data, valid_data

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        dataset_text_field='messages',
        formatting_func=lambda x: tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False),
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )

    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        dataset_text_field='messages',
        formatting_func=lambda x: tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False),
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )

    return train_dataset, valid_dataset


def create_datasets(tokenizer, args, seed=None):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )



    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=seed)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=seed)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


bnb_config = None
if script_args.use_bnb:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = "<|reserved_special_token_0|>" # don't use the eos token for padding during finetuning since it can stop the model from learning when to stop
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
 
dataset = load_from_disk("deli_friction_dataset")

train_dataset = dataset["train"]
eval_dataset = dataset["test"]
# Define the response template
response_template = "<|start_header_id|>assistant<|end_header_id|>"
instruction_template = "<|start_header_id|>user<|end_header_id|>"
collator = DataCollatorForFullAssistantLM(
    response_template=response_template,
    instruction_template=instruction_template,
    tokenizer=tokenizer
)

train_dataset, eval_dataset = generate_multiturn_dataset(
    tokenizer=tokenizer,
    args=script_args,
    train_data=train_dataset,
    valid_data=eval_dataset,
    seed=None
)
print("size of training and test datasets",train_dataset,eval_dataset  )


# Trainer
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=6096, # more tokens for multiturn
    tokenizer=tokenizer,
    args=training_args,
    data_collator=collator,
    packing=False
)

trainer.train()



trainer.save_model(training_args.output_dir)

output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)


def format_chat(example):
    formatted_chat = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    tokenized = tokenizer(
        formatted_chat,
        padding=False,           # Let the collator handle padding
        truncation=True,
    )
    
    return tokenized



