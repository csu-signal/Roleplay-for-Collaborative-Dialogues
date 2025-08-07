import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback
)
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_torch_fx_proxy
from accelerate import PartialState, Accelerator
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from peft import LoraConfig, PeftModel, get_peft_model
import bitsandbytes as bnb

import os
import torch
import pickle
from datasets import load_from_disk, DatasetDict
from transformers import (
   AutoModelForCausalLM, 
   AutoTokenizer,
   HfArgumentParser,
   set_seed
)
from peft import LoraConfig
from accelerate import Accelerator
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
import gc

from friction_trainer import frictionTrainer  # Import friction ++ trainer

@dataclass
class TrainingArguments:
   """Arguments for friction ++ training"""
   model_name: str = "sft_trained_weights" # add SFT model lora weights
   output_dir: str = "friction ++_outputs"
   data_path: str = "data"
   learning_rate: float = 5e-7
   batch_size: int = 8
   max_steps: int = 2000
   eval_steps: int = 100
   logging_steps: int = 10
   save_steps: int = 500
   max_length: int = 6096
   seed: int = 42

def main():
   # Parse arguments
   parser = HfArgumentParser(TrainingArguments)
   args = parser.parse_args_into_dataclasses()[0]
   
   # Set seed
   set_seed(args.seed)
   
   # Initialize accelerator and tokenizer
   accelerator = Accelerator()
   tokenizer = AutoTokenizer.from_pretrained(args.model_name)
   tokenizer.pad_token = tokenizer.eos_token
   tokenizer.padding_side = "right"

   # Load and process dataset
   dataset = load_from_disk(args.data_path)
   train_test_split = dataset["train"].train_test_split(test_size=0.05, seed=args.seed)
   dataset = DatasetDict({
       "train": train_test_split["train"],
       "test": train_test_split["test"]
   })

   # Define experiment configurations
   loss_types = {
       'friction ++': 5e-7,
       'friction ++_not_conditioned': 5e-7, 
       'friction ++_first_part_only': 5e-7
   }
   beta_values = [10, 5, 1, 0.01]
   experiment_logs = defaultdict(list)
   
   def run_experiment(model_path, beta, lr, loss_type, train_data, eval_data):
       """Run single friction ++ experiment"""
       # Initialize model
       model = AutoModelForCausalLM.from_pretrained(
           model_path,
           torch_dtype=torch.bfloat16,
           device_map={"": Accelerator().local_process_index}
       )
       model.config.use_cache = False
       
       # PEFT config
       peft_config = LoraConfig(
           r=8,
           lora_alpha=32,
           lora_dropout=0.05,
           target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "wte"],
           bias="none",
           task_type="CAUSAL_LM"
       )

       # Initialize trainer
       trainer = frictionTrainer(
           model=model,
           args=args,
           beta=beta,
           train_dataset=train_data,
           eval_dataset=eval_data,
           tokenizer=tokenizer,
           peft_config=peft_config
       )

       # Train
       trainer.train()
       
       # Save model and logs
       experiment_name = f"{loss_type}_{beta}"
       output_dir = os.path.join(args.output_dir, experiment_name)
       os.makedirs(output_dir, exist_ok=True)
       
       trainer.model.save_pretrained(output_dir)
       tokenizer.save_pretrained(output_dir)
       
       # Log results
       experiment_logs[loss_type].append({
           'hyperparameter': f"{beta}_{lr}",
           'logs': trainer.state.log_history
       })

       # Cleanup
       del model, trainer
       gc.collect()

   # Run experiments
   for loss_type, lr in tqdm(loss_types.items(), desc='Loss Types'):
       for beta in tqdm(beta_values, desc='Beta Values'):
           print(f"Running experiment: beta={beta}, lr={lr}, loss={loss_type}")
           run_experiment(
               args.model_name,
               beta,
               lr, 
               loss_type,
               dataset["train"],
               dataset["test"]
           )

   # Save logs
   with open('friction_logs/experiment_logs.pkl', 'wb') as f:
       pickle.dump(experiment_logs, f)

if __name__ == "__main__":
   main()

