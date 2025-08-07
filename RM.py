import pickle
import os
import torch
import torch.nn.functional as F
import inspect
import warnings
from collections import defaultdict
from dataclasses import FrozenInstanceError, replace, dataclass, field, asdict
from datasets import load_from_disk
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset,load_dataset, DatasetDict
from torch.utils.data import DataLoader
from peft import LoraConfig
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
import sys 
from trl.import_utils import is_peft_available
from trl import RewardConfig, RewardTrainer
from trl.trainer.utils import RewardDataCollatorWithPadding, compute_accuracy, print_rich_table
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer,  HfArgumentParser, set_seed
import warnings
from accelerate import PartialState
from tqdm import tqdm
from evaluate import load
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
import evaluate
import numpy as np
tqdm.pandas()
# Remove unwanted Jupyter arguments
if 'ipykernel_launcher' in sys.argv[0]:
    sys.argv = sys.argv[:1]


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    reward_model_type: Optional[str] = field(
        default="facebook/opt-1.3B",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=6, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    dataset: Optional[str] = field(default="delidata", metadata={"help": "the dataset used for training and evaluation "})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    dataset_num_proc: Optional[int] = field(default=24, metadata={"help": "the number of processes to preprocess and tokenize dataset"})
    max_steps: Optional[int] = field(default=1250, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=20, metadata={"help": "the evaluation frequency"})
    eval_first_step: Optional[bool] = field(default=False, metadata={"help": "whether to evaluate first step before training"})
         

    output_dir: Optional[str] = field(default="./friction_rm_alldata_results_delidata", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

def preprocess_function(examples):
    max_length = script_args.max_length  # Set the maximum length
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    # Statistics tracking
    lengths_chosen = []
    lengths_rejected = []
    truncated_chosen = 0
    truncated_rejected = 0

    # Iterate over the chosen and rejected examples
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        # Concatenate content from list of dicts for 'chosen' and 'rejected'
        if isinstance(chosen, list):
            chosen_text = []
            for entry in chosen:
                content = entry.get('content', '')
                if entry.get('role') == 'assistant':
                    content = f"</s> {content} </s>"  # Add </s> before and after the assistant's response
                chosen_text.append(content)
            chosen = ' '.join(chosen_text)

        if isinstance(rejected, list):
            rejected_text = []
            for entry in rejected:
                content = entry.get('content', '')
                if entry.get('role') == 'assistant':
                    content = f"</s> {content} </s>"  # Add </s> before and after the assistant's response
                rejected_text.append(content)
            rejected = ' '.join(rejected_text)

        # Tokenize the chosen and rejected examples
        tokenized_chosen = tokenizer(chosen, truncation=True, max_length=max_length, padding="max_length")
        tokenized_rejected = tokenizer(rejected, truncation=True, max_length=max_length, padding="max_length")

        # Track lengths
        lengths_chosen.append(len(tokenizer(chosen)["input_ids"]))
        lengths_rejected.append(len(tokenizer(rejected)["input_ids"]))

        # Check if truncated
        if len(tokenizer(chosen)["input_ids"]) > max_length:
            truncated_chosen += 1
        if len(tokenizer(rejected)["input_ids"]) > max_length:
            truncated_rejected += 1

        # Append tokenized results to the new_examples dictionary
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    # Print statistics
    print(f"Chosen Texts: Min Length = {min(lengths_chosen)}, Max Length = {max(lengths_chosen)}, Mean Length = {sum(lengths_chosen) / len(lengths_chosen)}")
    print(f"Rejected Texts: Min Length = {min(lengths_rejected)}, Max Length = {max(lengths_rejected)}, Mean Length = {sum(lengths_rejected) / len(lengths_rejected)}")
    print(f"Truncated Chosen: {truncated_chosen} / {len(lengths_chosen)}")
    print(f"Truncated Rejected: {truncated_rejected} / {len(lengths_rejected)}")

    return new_examples



def transform_and_assign_preferences_wtd(row):
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

    chosen_rationale =  row["chosen_rationale"]
    rejected_rationale = row["rejected_rationale"]
    chosen_response = [
        {'content': prompt, 'role': 'user'},
        {'content': chosen_rationale + "." + row["chosen_friction_statement"].replace('\n', ' '), 'role': 'assistant'}
    ]

    # Format the rejected response
    rejected_response = [
        {'content': prompt, 'role': 'user'},
        {'content': rejected_rationale + "." + row["rejected_friction_statement"].replace('\n', ' '), 'role': 'assistant'}
    ]

    # Return the new structure with feedback weights
    return {
        'prompt': prompt,
        'chosen': chosen_response,
        'rejected': rejected_response,
    }

def transform_and_assign_preferences_deli(row):
    system_prompt_rm = (
    "Please rate the following friction intervention in the context of the **dialogue history** from the Wason Card Selection Task. "
    "Participants see four cards with numbers or letters and must test the rule: "
    "'All cards with vowels on one side must have an even number on the other.'"
    "A friction intervention is a statement designed to act as indirect persuasion, prompting participants to "
    "reevaluate their assumptions when they misinterpret the rule or overlook counterexamples."
)

    prompt = (system_prompt_rm + row['context']).replace('\n', ' ')

    chosen_rationale =  row["chosen_rationale"]
    rejected_rationale = row["rejected_rationale"]
    chosen_response = [
        {'content': prompt, 'role': 'user'},
        {'content': chosen_rationale + "." + row["chosen_friction"].replace('\n', ' '), 'role': 'assistant'}
    ]

    # Format the rejected response
    rejected_response = [
        {'content': prompt, 'role': 'user'},
        {'content': rejected_rationale + "." + row["rejected_friction"].replace('\n', ' '), 'role': 'assistant'}
    ]

    # Return the new structure with feedback weights
    return {
        'prompt': prompt,
        'chosen': chosen_response,
        'rejected': rejected_response,
    }

if __name__ == "__main__":

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)
    script_args_dict = asdict(script_args)

    config = RewardConfig(
     
    output_dir=script_args.output_dir,
    per_device_train_batch_size=4,
    per_device_eval_batch_size = 4,
    # num_train_epochs=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=1e-6,
    report_to="wandb",
    remove_unused_columns=False,
    optim="adamw_torch",
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    max_length=1024,
    max_steps = 12000,
    save_steps = 3000
  
 
)
    reward_model_type = script_args.reward_model_type  

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configuration for gradient checkpointing
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    config.max_prompt_length = 128
    config.max_target_length = 512
    config.per_device_eval_batch_size = 4

    config._n_gpu = 1
    print("config max length", config.max_prompt_length,config.max_target_length) 
    print("config per_device_eval_batch_size", config.per_device_eval_batch_size, config._n_gpu  ) 
    ################
    # Model & Tokenizer
    ################

    # Set the torch_dtype based on the argument (e.g., float16 or bfloat16)
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32  # Default to float32 if not specified
        
    torch_dtype = torch.bfloat16
 
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        reward_model_type, 
        trust_remote_code=True, 
        use_fast=True
    )
 
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_type,
        num_labels=1,  # Set to 1 for reward model classification
        trust_remote_code=True,
        torch_dtype=torch_dtype,
 
    ).to(device)
 
    print(f"Model {reward_model_type} loaded on device: {device}")
    print(f"Tokenizer for {reward_model_type} loaded successfully.")
    print(f"Config for {config} loaded successfully.")

    
    # Define the metric that we'll use for validation.
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        rewards_chosen, rewards_rejected = eval_pred
        predictions, _ = eval_pred
        predictions = np.argmax(predictions, axis=0)
        labels = np.zeros(predictions.shape)
        original_acc = accuracy_metric.compute(predictions=predictions, references=labels)
        
        # Check if rewards_chosen is 2D and simplify (use max or mean)
        if rewards_chosen.ndim > 1:
            rewards_chosen = rewards_chosen.max(axis=-1)  # You can use mean() if more appropriate

        # Flatten any extra dimensions
        rewards_chosen = rewards_chosen.squeeze()
        rewards_rejected = rewards_rejected.squeeze()

        # Compute accuracy: how often is the chosen reward higher than the rejected reward?
        correct = (rewards_chosen > rewards_rejected).astype(int)

        # Calculate accuracy as the mean of correct comparisons
        accuracy = correct.mean().item()

        # Compute the reward margin (how much better is chosen vs rejected)
        reward_margin = (rewards_chosen - rewards_rejected).mean().item()

        return {
            "accuracy": accuracy,
            "reward_margin": reward_margin,
            "original_acc":original_acc['accuracy'] 
        }



    class OPTRewardTrainer(RewardTrainer):
        r"""
        OPTRewardTrainer is a subclass of the RewardTrainer that is specifically designed for training rewards with the generic RM (log-sigmoid of reward difference) from ORPO paper 
        Adapted from https://arxiv.org/pdf/2406.10216 where they use 
        """

        def __init__(
            self,
            script_args: Dict[str, Union[str, float, int, bool, Dict]],
            model: Optional[Union[PreTrainedModel, nn.Module]] = None,
            teacher_model:Optional[Union[PreTrainedModel, nn.Module]] = None,  
            args: Optional[RewardConfig] = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            teacher_tokenizer:Optional[PreTrainedTokenizerBase] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
                None,
                None,
            ),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            max_length: Optional[int] = None,
            peft_config: Optional[Dict] = None,

            reward_scaling_factor_alpha : float = 0.01,  # New argument for scaling dense rewards,
            reward_kl_beta  = 0.1,  # KL regularization factor,
            is_encoder_decoder = None,
            beta: float = 0.1,
            label_smoothing: float = 0,
            loss_type: Optional[str] = None,
            truncation_mode: str = "keep_end",
            label_pad_token_id: int = -100,

            max_prompt_length: Optional[int] = None,
            max_target_length: Optional[int] = None,
  
        ):
            """
            Initialize OPTRewardTrainer.

            Args:
                reward_scaling_factor (`float`, defaults to `1.0`):
                    A scaling factor to apply to the dense rewards, allowing for easier handling of different reward distributions.
                All other arguments are inherited from the RewardTrainer.
            """


            super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                model_init=model_init,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                optimizers=optimizers,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                max_length=max_length,
                peft_config=peft_config,

            )
            # self.teacher_model = teacher_model
            self.model = model
            self.tokenizer = tokenizer
            self.reward_scaling_factor_alpha = reward_scaling_factor_alpha
            self.reward_kl_beta = reward_kl_beta
            self.beta = beta
            self.label_smoothing = label_smoothing
            self.loss_type = loss_type
            self.truncation_mode =truncation_mode
            self.label_pad_token_id = label_pad_token_id
            self.max_prompt_length = args.max_prompt_length
            self.max_target_length = args.max_target_length
            self.is_encoder_decoder = False #boolean is true if you want to use enc decoder model instead of decoder only 
            self.is_vision_model = False
            self._stored_metrics = defaultdict(lambda: defaultdict(list))
            self._stored_sample_level_metrics = defaultdict(lambda: defaultdict(list))
            self.script_args = script_args

        def get_batch_loss_metrics_new(self, batch_metrics):

            metrics = {}
            rewards_chosen, rewards_rejected, loss = batch_metrics
            prefix = "train_"
            metrics[f"{prefix}rewards/chosen"] = rewards_chosen.mean().cpu()
            metrics[f"{prefix}rewards/rejected"] = rewards_rejected.mean().cpu()

            reward_accuracies = (rewards_chosen > rewards_rejected).float()

            metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
            metrics[f"{prefix}rewards/margins"] = (rewards_chosen - rewards_rejected).mean().cpu()
 
            return metrics
        
        # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://huggingface.co/papers/2203.02155
        def compute_loss(self, model, inputs, return_outputs=False, train_eval= None,):
            
            metrics = {}
            sample_level_metrics = {}
            rewards_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])[0]
            rewards_rejected = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"])[0]
            loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
#    
            if train_eval:
    
                prefix = "eval_" 
                metrics[f"{prefix}rewards/chosen"] = rewards_chosen.mean().cpu()
                metrics[f"{prefix}rewards/rejected"] = rewards_rejected.mean().cpu()

                reward_accuracies = (rewards_chosen > rewards_rejected).float()

                metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
                metrics[f"{prefix}rewards/margins"] = (rewards_chosen - rewards_rejected).mean().cpu()
                sample_level_metrics[f"{prefix}rewards/accuracies_sample_level"] = reward_accuracies.cpu()
                # print("sample_level_metrics acc",reward_accuracies.cpu() )
                sample_level_metrics[f"{prefix}rewards/margin_sample_levels"] = (rewards_chosen - rewards_rejected).cpu()
                # print("sample_level_metrics margin", (rewards_chosen - rewards_rejected).cpu())
                reward_accuracies = (rewards_chosen > rewards_rejected).float()
 

                self.store_metrics(metrics, train_eval="eval")
                self.store_sample_level_metrics(sample_level_metrics, train_eval="eval")
                
 
            else:
                batch_metrics = (rewards_chosen, rewards_rejected, loss)
                metrics = self.get_batch_loss_metrics_new(batch_metrics)
                self.store_metrics(metrics, train_eval="eval")
            if return_outputs:
                return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
            return loss

        def prediction_step(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            with torch.no_grad():
                print("shape before torch no grad compute loss input_ids_chosen", inputs["input_ids_chosen"].size())
                print("shape before torch no grad compute loss input_ids_rejected", inputs["input_ids_rejected"].size())
                loss, logits_dict  = self.compute_loss(model, inputs, return_outputs=True, train_eval = "eval")

            if prediction_loss_only:
                return (loss, None, None)

            loss = loss.detach()
            logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
            logits = nested_detach(logits)
            logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T
            labels = torch.zeros(logits.shape[0])
            labels = self._prepare_inputs(labels)

            return loss, logits, labels


        def evaluate(self, *args, **kwargs):
            num_print_samples = kwargs.pop("num_print_samples", 2)
            # self.visualize_samples(num_print_samples)
            return super().evaluate(*args, **kwargs)


        def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
            for key, value in metrics.items():
                self._stored_metrics[train_eval][key].append(value)

        def store_sample_level_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
            for key, value in metrics.items():
                # Initialize nested dictionary structure if not already present
                if key not in self._stored_sample_level_metrics[train_eval]:
                    self._stored_sample_level_metrics[train_eval][key] = {}

                if self.state.global_step not in self._stored_sample_level_metrics[train_eval][key]:
                    self._stored_sample_level_metrics[train_eval][key][self.state.global_step] = []

                # Append the value to the list for the current global step
                self._stored_sample_level_metrics[train_eval][key][self.state.global_step].append(value)
 

        def sanitize_logit_values(self, logits):
        # Replace NaN values with a valid placeholder (e.g., 0, None, or 'nan')
            return [[0 if np.isnan(inner_item) else inner_item for inner_item in item] for item in logits]
        
        def log(self, logs: Dict[str, float]) -> None:
            """
            Log `logs` on the various objects watching training, including stored metrics.

            Args:
                logs (`Dict[str, float]`):
                    The values to log.
            """
            # logs either has 'loss' or 'eval_loss'
            train_eval = "train" if "loss" in logs else "eval"
            # Add averaged stored metrics to logs
            for key, metrics in self._stored_metrics[train_eval].items():
                logs[key] = torch.tensor(metrics).mean().item()
            # Save _stored_metrics as a pickle file
            metrics_file = f"friction_deli_rm_files/stored_metrics_{train_eval}_basic_rm.pkl"
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

            with open(metrics_file, 'wb') as f:
                pickle.dump(self._stored_sample_level_metrics[train_eval], f)
            del self._stored_metrics[train_eval]
            return super().log(logs)

        def visualize_samples(self, num_print_samples: int):
            """
            Visualize the reward model logits prediction

            Args:
                num_print_samples (`int`, defaults to `4`):
                    The number of samples to print. Set to `-1` to print all samples.
            """
            eval_dataloader = self.get_eval_dataloader()
            table = defaultdict(list)

            for _, inputs in enumerate(eval_dataloader):
                _, logits, _ = self.prediction_step(self.model, inputs, prediction_loss_only=False)
                chosen_text = self.tokenizer.batch_decode(inputs["input_ids_chosen"], skip_special_tokens=True)
                rejected_text = self.tokenizer.batch_decode(inputs["input_ids_rejected"], skip_special_tokens=True)

                # Gather text and logits
                table["chosen_text"].extend(gather_object(chosen_text))
                table["rejected_text"].extend(gather_object(rejected_text))

                # Sanitize logits by replacing NaN values with 0 (or another placeholder)
                sanitized_logits = self.sanitize_logit_values(logits.tolist())
                table["logits"].extend(gather_object([[round(inner_item, 4) for inner_item in item] for item in sanitized_logits]))

                # Break if we've reached the desired number of print samples
                if num_print_samples >= 0 and len(table["chosen_text"]) >= num_print_samples:
                    break

            # Convert table to a dataframe
            df = pd.DataFrame(table)

            # Print and log results
            if self.accelerator.process_index == 0:
#                 print_rich_table(df[:num_print_samples])
                if "wandb" in self.args.report_to:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"completions": wandb.Table(dataframe=df)})

    
 
    if script_args.dataset == "wtd":
        print(f"processing {script_args.dataset} dataset", tokenizer)

        friction_data = load_from_disk("##")
        friction_data_2 = load_from_disk("##")  
        # Convert the datasets to pandas DataFrames
        train_df_1 = pd.DataFrame(friction_data['train'])
        test_df_1 = pd.DataFrame(friction_data['test'])
        train_df_2 = pd.DataFrame(friction_data_2['train'])

        # Concatenate train and test sets for both datasets
        friction_df_1 = pd.concat([train_df_1, test_df_1], ignore_index=True)
        friction_df_2 = train_df_2.copy()

        # Rename columns in the second friction dataset
        columns_to_rename = {
            'chosen_context': 'context',
            'chosen_task_state': 'task_state',
            'chosen_belief_state': 'belief_state'

        }
        friction_df_2.rename(columns=columns_to_rename, inplace=True)

        # Concatenate the two datasets
        combined_df = pd.concat([friction_df_1, friction_df_2], ignore_index=True)

        # Convert the combined DataFrame back to a Hugging Face Dataset
        combined_dataset = Dataset.from_pandas(combined_df)

        # Create a DatasetDict with a single split for the combined dataset
        final_dataset = DatasetDict({'combined': combined_dataset})

        # Print a sample from the combined dataset
        print(final_dataset['combined'][0])
        split_dataset = final_dataset['combined'].train_test_split(test_size=0.05, shuffle=True, seed=42)
        friction_data = DatasetDict({
                'train': split_dataset['train'],
                'test': split_dataset['test']
            })
        # Apply the transformation to both train and test datasets in a single pass
        train_wtd = friction_data['train'].map(transform_and_assign_preferences_wtd)
        test_wtd = friction_data['test'].map(transform_and_assign_preferences_wtd)
 
        # for sanity checking
        dummy_train_dataset = Dataset.from_dict({'prompt': train_wtd['prompt'][:100], 'chosen': train_wtd['chosen'][:100], 
        'rejected': train_wtd['rejected'][:100] })
        dummy_eval_dataset = Dataset.from_dict({'prompt': test_wtd['prompt'][:12], 'chosen': test_wtd['chosen'][:12], 
        'rejected': test_wtd['rejected'][:12] })

        print("size of dummy_train_dataset and dummy_eval_dataset dataset", dummy_train_dataset, dummy_eval_dataset)
        
        with PartialState().local_main_process_first():
            train_data = train_wtd.map(
                preprocess_function,
                batched=True,
                num_proc=24,
            )

            eval_data = test_wtd.map(
                preprocess_function,
                batched=True,
                num_proc=24,
            )
        print("size of train and eval dataset", train_data , eval_data )

    elif script_args.dataset == "delidata":
        print("getting dataset", script_args.dataset)
        friction_data = load_from_disk("##") #training the SFT model on chosen friction for deli data
        friction_data_test = load_from_disk("##")
        # Separate into train and test sets
        train_data_deli = friction_data['train'].map(transform_and_assign_preferences_deli)
        valid_data_deli = friction_data_test['test'].map(transform_and_assign_preferences_deli)

        dummy_train_dataset = Dataset.from_dict({'prompt': train_data_deli['prompt'][:100], 'chosen': train_data_deli['chosen'][:100], 
        'rejected': train_data_deli['rejected'][:100] })
        dummy_eval_dataset = Dataset.from_dict({'prompt': valid_data_deli['prompt'][:12], 'chosen': valid_data_deli['chosen'][:12], 
        'rejected': valid_data_deli['rejected'][:12] })

        print("size of dummy_train_dataset and dummy_eval_dataset dataset", dummy_train_dataset, dummy_eval_dataset)
         
        with PartialState().local_main_process_first():
            train_data = train_data_deli.map(
                preprocess_function,
                batched=True,
                num_proc=24,
            )

            eval_data = valid_data_deli.map(
                preprocess_function,
                batched=True,
                num_proc=24,
            )
        print("size of train and eval dataset", train_data , eval_data )

    trainer = OPTRewardTrainer(
        script_args = script_args_dict,
        model=model,
        teacher_model = None, 
        tokenizer = tokenizer, 
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
        # peft_config=peft_config,

        # data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    trainer.model.save_pretrained(config.output_dir)
    # trainer.push_to_hub()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)

        # Access the trainer's state history
    trainer_state_history = trainer.state.log_history
   

    # Save the trainer's state history to a pickle file
    with open("friction_deli_rm_files/trainer_state_log_history.pkl", "wb") as f:
        pickle.dump(trainer_state_history, f)




