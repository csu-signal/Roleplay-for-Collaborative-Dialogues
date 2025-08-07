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


 
class frictionTrainer(DPOTrainer):
    """
    A subclass of DPOTrainer that inherits everything from the parent class
    and adds custom tokenization for datasets.
    """
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, torch.nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, torch.nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo","friction",  "friction_not_conditioned", "friction_first_part_only", "aot", "aot_pair"] = "friction",
        args: Optional[DPOTrainer] = None,
        data_collator: Optional[Callable] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,
    ):
        # Call the parent class constructor first
        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type=loss_type,
            args=args,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            peft_config=peft_config,
            is_encoder_decoder=is_encoder_decoder,
            disable_dropout=disable_dropout,
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
            precompute_ref_log_probs=precompute_ref_log_probs,
            dataset_num_proc=dataset_num_proc,
            model_init_kwargs=model_init_kwargs,
            ref_model_init_kwargs=ref_model_init_kwargs,
            model_adapter_name=model_adapter_name,
            ref_adapter_name=ref_adapter_name,
            reference_free=reference_free,
            force_use_ref_model=force_use_ref_model,
        )

        # # Now we can use self.dataset_num_proc after super() call
        # with PartialState().local_main_process_first():
        #     # num_proc=min(4, self.dataset_num_proc)  # Lower value
        #     if train_dataset is not None:
        #         self.train_dataset = train_dataset.map(
        #             self.tokenize_row,
        #             num_proc=self.dataset_num_proc,
        #             writer_batch_size=10
        #         )
        #         print("Train dataset tokenized in frictionTrainer subclass", self.train_dataset )

        #     if eval_dataset is not None:
        #         self.eval_dataset = eval_dataset.map(
        #             self.tokenize_row,
        #             num_proc=self.dataset_num_proc,
        #             writer_batch_size=10
        #         )
        #         print("Eval dataset tokenized in frictionTrainer subclass", self.eval_dataset )

        # Store additional attributes
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.rouge_scores_list = []  # Persistent state to accumulate ROUGE scores
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)  # Initialize RougeScorer
        self.bleu_metric = load_metric("bleu")
        self.meteor_metric = load_metric("meteor")
        self.sentence_transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.loss_type = loss_type

    
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_chosen_logp, reference_rejected_logp = self.compute_reference_log_probs(padded_batch)
                reference_chosen_logp, reference_rejected_logp = self.accelerator.gather_for_metrics(
                    (reference_chosen_logp, reference_rejected_logp)
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_chosen_logps).float().numpy()
            all_reference_rejected_logps = torch.cat(reference_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="reference_chosen_logps", column=all_reference_chosen_logps)
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def extract_task_and_beliefs(self, chosen_text):
        """
        Extracts everything before '### Assistant:', keeps '### Assistant:', 
        removes <rationale> ... </rationale> and <friction> ... </friction>,
        keeps <t> ... </t> and <b> ... </b>, and ensures <|eot_id|> is retained.

        :param chosen_text: The full chosen response string.
        :return: Extracted and formatted task and belief segment.
        """

        # Step 1: Capture everything before '### Assistant:' and include '### Assistant:'
        match = re.search(r"(.*?)(### Assistant:.*?)\s*", chosen_text, re.DOTALL)

        if not match:
            return ""  # Return empty string if '### Assistant:' is not found

        full_prefix = match.group(1).strip()  # Everything before '### Assistant:'
        assistant_part = match.group(2).strip()  # '### Assistant:' marker included
        print("assistant_part", assistant_part)
        # Step 2: Extract <t> ... </t> and <b> ... </b>
        task_belief_match = re.search(r"(<t>.*?</b>)", chosen_text, re.DOTALL)

        if not task_belief_match:
            return ""  # If task/belief section is missing, return empty

        task_belief_part = task_belief_match.group(1).strip()  # Keep the task & belief segment
        print("task_belief_part", task_belief_part)

        # Step 3: Extract trailing <|eot_id|>
        eot_match = re.search(r"(<\|eot_id\|>.*)$", chosen_text, re.DOTALL)

        trailing_content ="<|eot_id|>"

        # Step 4: Remove <rationale> ... </rationale> and <friction> ... </friction>
        chosen_cleaned = re.sub(r"<rationale>.*?</rationale>", "", chosen_text, flags=re.DOTALL)
        chosen_cleaned = re.sub(r"<friction>.*?</friction>", "", chosen_cleaned, flags=re.DOTALL)
        print("chosen_cleaned", chosen_cleaned)
        # Step 5: Construct the final parsed output
        chosen_task_beliefs = f"{full_prefix}\n\n{assistant_part} {task_belief_part}\n{trailing_content}"
        print("full_prefix", full_prefix)
        print("trailing_content", trailing_content)
        return chosen_task_beliefs


    def extract_chosen_friction(self, chosen_text):
        """
        Extracts everything before '### Assistant:', keeps '### Assistant:',
        removes <t> ... </t> and <b> ... </b>,
        and retains only the SECOND occurrence of <rationale> ... </friction> along with the trailing <|eot_id|>.

        :param chosen_text: The full chosen response string.
        :return: Extracted and formatted response.
        """

        # Step 1: Capture everything before '### Assistant:' and include '### Assistant:'
        match = re.search(r"(.*?)(### Assistant:.*?)\s*", chosen_text, re.DOTALL)

        if not match:
            return ""  # Return empty string if '### Assistant:' is not found

        full_prefix = match.group(1).strip()  # Everything before '### Assistant:'
        assistant_part = match.group(2).strip()  # '### Assistant:' marker included

        # Step 2: Find all occurrences of <rationale> ... </friction>
        rationale_friction_matches = list(re.finditer(r"(<rationale>.*?</friction>)", chosen_text, re.DOTALL))

        if len(rationale_friction_matches) < 2:
            return ""  # If there isn't a second occurrence, return empty

        rationale_friction_part = rationale_friction_matches[1].group(1).strip()  # Capture SECOND occurrence

        # Step 3: Capture everything after the second occurrence, including <|eot_id|>
        trailing_match = re.search(rf"{re.escape(rationale_friction_part)}(.*?)<\|eot_id\|>", chosen_text, re.DOTALL)

        if not trailing_match:
            return ""  # Ensure we correctly capture the end

        trailing_content = rationale_friction_part + trailing_match.group(1).strip() + "\n<|eot_id|>"

        # Step 4: Remove <t> ... </t> and <b> ... </b> including tags and their content
        assistant_cleaned = re.sub(r"<t>.*?</t>", "", assistant_part, flags=re.DOTALL)
        assistant_cleaned = re.sub(r"<b>.*?</b>", "", assistant_cleaned, flags=re.DOTALL)

        # Step 5: Construct the final parsed output
        parsed_output = f"{full_prefix}\n\n{assistant_cleaned} {trailing_content}"

        return parsed_output

    def build_tokenized_answer(self, prompt, answer, images=None):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """
        if self.is_vision_model:
            if answer.count("<image>") > 0:
                raise NotImplementedError("Answer contains <image> token, which is not supported yet.")
            if "add_special_tokens" in inspect.signature(self.processor).parameters:
                processor_kwargs = {"add_special_tokens": False}
            else:
                processor_kwargs = {}
            full_tokenized = self.processor(prompt + answer, images=images, **processor_kwargs)
            full_tokenized = {k: v[0] for k, v in full_tokenized.items()}  # Unbatch, not done when using idefics
            if not isinstance(full_tokenized["input_ids"], list):  # llava processor returns tensors
                full_tokenized["input_ids"] = full_tokenized["input_ids"].tolist()
                full_tokenized["attention_mask"] = full_tokenized["attention_mask"].tolist()
            prompt_input_ids = self.processor(prompt, images=images, **processor_kwargs)["input_ids"][0]
            if not isinstance(prompt_input_ids, list):  # llava processor returns tensors
                prompt_input_ids = prompt_input_ids.tolist()
        else:
            full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
            prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return_dict = dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )
        if "pixel_values" in full_tokenized:
            return_dict["prompt_pixel_values"] = full_tokenized["pixel_values"]
        if "pixel_attention_mask" in full_tokenized:
            return_dict["prompt_pixel_attention_mask"] = full_tokenized["pixel_attention_mask"]

        return return_dict


    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """
        Per row tokenizer, customized from tokenize_row of DPO trainer class. 
        Additionally, tokenizes the friction (both chosen and rejected) interventions without the frictive "phi" (belief) state labels. 
        These input labels are used to compute both the phi-conditioned logits and the phi-unconditioned logits. 
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]
        
        # New friction versions
        chosen_friction = self.extract_chosen_friction(chosen)
        rejected_friction = self.extract_chosen_friction(rejected)
        images = feature.get("images")

        if not self.is_encoder_decoder:
            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")

            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            # Tokenize all responses (original and friction)
            chosen_tokens = self.build_tokenized_answer(prompt, chosen, images)
            rejected_tokens = self.build_tokenized_answer(prompt, rejected, images)
            chosen_friction_tokens = self.build_tokenized_answer(prompt, chosen_friction, images)
            rejected_friction_tokens = self.build_tokenized_answer(prompt, rejected_friction, images)

            # Get lengths
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])
            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            chosen_friction_prompt_len_input_ids = len(chosen_friction_tokens["prompt_input_ids"])
            rejected_friction_prompt_len_input_ids = len(rejected_friction_tokens["prompt_input_ids"])
            
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids,
                                    chosen_friction_prompt_len_input_ids, rejected_friction_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Add BOS token to all sequences
            bos_token_id = self.tokenizer.bos_token_id
            for tokens, length in [(prompt_tokens, prompt_len_input_ids),
                                (chosen_tokens, chosen_prompt_len_input_ids),
                                (rejected_tokens, rejected_prompt_len_input_ids),
                                (chosen_friction_tokens, chosen_friction_prompt_len_input_ids),
                                (rejected_friction_tokens, rejected_friction_prompt_len_input_ids)]:
                if length == 0 or bos_token_id != tokens["prompt_input_ids"][0]:
                    tokens["prompt_input_ids"] = [bos_token_id] + tokens["prompt_input_ids"]
                    tokens["prompt_attention_mask"] = [1] + tokens["prompt_attention_mask"]

            # Add EOS token to all responses
            eos_token_id = self.tokenizer.eos_token_id
            for tokens in [chosen_tokens, rejected_tokens, chosen_friction_tokens, rejected_friction_tokens]:
                if len(tokens["input_ids"]) == 0 or eos_token_id != tokens["input_ids"][-1]:
                    tokens["input_ids"].append(eos_token_id)
                    tokens["attention_mask"].append(1)

            # Handle length truncation
            longer_response_length = max(len(chosen_tokens["input_ids"]), 
                                    len(rejected_tokens["input_ids"]),
                                    len(chosen_friction_tokens["input_ids"]),
                                    len(rejected_friction_tokens["input_ids"]))

            # Truncate prompts if needed
            for answer_tokens in [chosen_tokens, rejected_tokens, chosen_friction_tokens, 
                                rejected_friction_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][:self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length:]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # Create sequence tokens for all responses
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_friction_sequence_tokens = {
                k: chosen_friction_tokens[f"prompt_{k}"] + chosen_friction_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_friction_sequence_tokens = {
                k: rejected_friction_tokens[f"prompt_{k}"] + rejected_friction_tokens[k] for k in ["input_ids", "attention_mask"]
            }

            # Create labels for all sequences
            for seq_tokens, tok in [(chosen_sequence_tokens, chosen_tokens),
                                (rejected_sequence_tokens, rejected_tokens),
                                (chosen_friction_sequence_tokens, chosen_friction_tokens),
                                (rejected_friction_sequence_tokens, rejected_friction_tokens)]:
                seq_tokens["labels"] = seq_tokens["input_ids"][:]
                seq_tokens["labels"][:len(tok["prompt_input_ids"])] = [self.label_pad_token_id] * len(tok["prompt_input_ids"])

            # Add everything to batch
            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "chosen_friction_": chosen_friction_sequence_tokens,
                "rejected_friction_": rejected_friction_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens



                        # Add these print statements just before returning the batch:
            # print("\n=== Verifying shapes of all fields ===")

            # # Original fields
            # print("\nOriginal Fields:")
            # print(f"chosen_input_ids shape: {len(batch['chosen_input_ids'])}")
            # print(f"chosen_attention_mask shape: {len(batch['chosen_attention_mask'])}")
            # print(f"chosen_labels shape: {len(batch['chosen_labels'])}")
            # print(f"rejected_input_ids shape: {len(batch['rejected_input_ids'])}")
            # print(f"rejected_attention_mask shape: {len(batch['rejected_attention_mask'])}")
            # print(f"rejected_labels shape: {len(batch['rejected_labels'])}")

            # # Friction fields
            # print("\nFriction Fields:")
            # print(f"chosen_friction_input_ids shape: {len(batch['chosen_friction_input_ids'])}")
            # print(f"chosen_friction_attention_mask shape: {len(batch['chosen_friction_attention_mask'])}")
            # print(f"chosen_friction_labels shape: {len(batch['chosen_friction_labels'])}")
            # print(f"rejected_friction_input_ids shape: {len(batch['rejected_friction_input_ids'])}")
            # print(f"rejected_friction_attention_mask shape: {len(batch['rejected_friction_attention_mask'])}")
            # print(f"rejected_friction_labels shape: {len(batch['rejected_friction_labels'])}")


        return batch

    # Add your custom function

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        
        def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat([tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)

        concatenated_batch = {}
        
        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], 
                            batch["rejected_labels"].shape[1],
                            batch["chosen_friction_labels"].shape[1], 
                            batch["rejected_friction_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], 
                            batch["rejected_input_ids"].shape[1],
                            batch["chosen_friction_input_ids"].shape[1], 
                            batch["rejected_friction_input_ids"].shape[1])

        # First process original chosen/rejected pairs
        for k in batch:
            if k.startswith("chosen") and not k.startswith("chosen_friction") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

        # Add original rejected
        for k in batch:
            if k.startswith("rejected") and not k.startswith("rejected_friction") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value)),
                    dim=0
                )

        # Now add friction pairs in the same way
        for k in batch:
            if k.startswith("chosen_friction") and isinstance(batch[k], torch.Tensor):
                concatenated_key = k.replace("chosen_friction", "concatenated")
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_batch[concatenated_key] = torch.cat(
                    (concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value)),
                    dim=0
                )

        # Finally add rejected friction
        for k in batch:
            if k.startswith("rejected_friction") and isinstance(batch[k], torch.Tensor):
                concatenated_key = k.replace("rejected_friction", "concatenated")
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_batch[concatenated_key] = torch.cat(
                    (concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value)),
                    dim=0
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(4, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(4, 1).to(device=device)



        return concatenated_batch


    def friction_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,      # π_θ(f_w|φ,x)
        policy_rejected_logps: torch.FloatTensor,     # π_θ(f_l|φ,x)
        reference_chosen_logps: torch.FloatTensor,    # π_ref(f_w|φ,x)
        reference_rejected_logps: torch.FloatTensor,  # π_ref(f_l|φ,x)
        policy_chosen_friction_logps: torch.FloatTensor,      # π_θ(f_w|x)
        policy_rejected_friction_logps: torch.FloatTensor,    # π_θ(f_l|x)
        reference_chosen_friction_logps: torch.FloatTensor,   # π_ref(f_w|x)
        reference_rejected_friction_logps: torch.FloatTensor, # π_ref(f_l|x)
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the friction loss combining conditional and unconditioned policy ratios for preference learning. Adapted from DPO loss in DPO trainer

        Args:
            policy_chosen_logps: Log probs of winning friction (f_w) under π_θ(·|φ,x). Shape: (batch_size,)
            policy_rejected_logps: Log probs of losing friction (f_l) under π_θ(·|φ,x). Shape: (batch_size,)
            reference_chosen_logps: Log probs of f_w under π_ref(·|φ,x). Shape: (batch_size,)
            reference_rejected_logps: Log probs of f_l under π_ref(·|φ,x). Shape: (batch_size,)
            policy_chosen_friction_logps: Log probs of f_w under π_θ(·|x). Shape: (batch_size,)
            policy_rejected_friction_logps: Log probs of f_l under π_θ(·|x). Shape: (batch_size,)
            reference_chosen_friction_logps: Log probs of f_w under π_ref(·|x). Shape: (batch_size,)
            reference_rejected_friction_logps: Log probs of f_l under π_ref(·|x). Shape: (batch_size,)

        Returns:
            A tuple of tensors: (losses, chosen_rewards, rejected_rewards, chosen_friction_rewards, rejected_friction_rewards)
            - losses: friction loss combining conditional and unconditioned policy differences
            - chosen_rewards: Rewards for winning frictions under conditional policy 
            - rejected_rewards: Rewards for losing frictions under conditional policy
            - chosen_friction_rewards: Rewards for winning frictions under unconditioned policy
            - rejected_friction_rewards: Rewards for losing frictions under unconditioned policy
            All tensors have shape (batch_size,)
        """
        
     
        chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps.to(self.accelerator.device)
        rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected_logps.to(self.accelerator.device)



        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            pi_friction_logratios = policy_chosen_friction_logps - policy_rejected_friction_logps

            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps
                ref_friction_logratios = reference_chosen_friction_logps - reference_rejected_friction_logps

            pi_logratios = pi_logratios.to(self.accelerator.device)
            pi_friction_logratios = pi_friction_logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            ref_friction_logratios = ref_friction_logratios.to(self.accelerator.device)
            logits = pi_logratios - ref_logratios
            # get the direct logits for the friction model
            friction_only_logits = pi_friction_logratios - ref_friction_logratios #second reward term in friction loss objective
            # print("printing diveergence type", self.f_divergence_type, self.loss_type)

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
       
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "friction":
            print("getting loss friction ++")
            # eqn (17) of the paper where beta is the regularization parameter for the friction loss, denoted by beta in the paper.
            # losses = (logits - 1 / ( * self.beta)) ** 2
            losses = (self.beta*(logits + friction_only_logits) - 1)** 2

            elif self.loss_type == "friction_not_conditioned":
            print("getting loss friction delta R") # delta R baseline in paper
            # eqn (17) of the paper where beta is the regularization parameter for the friction loss, denoted by beta in the paper.
            # losses = (logits - 1 / ( * self.beta)) ** 2
            losses = (self.beta*(friction_only_logits) - 1)** 2

        elif self.loss_type == "friction_first_part_only":
            print("getting loss friction")
            # eqn (17) of the paper where beta is the regularization parameter for the friction loss, denoted by beta in the paper.
            # losses = (logits - 1 / ( * self.beta)) ** 2
            losses = (self.beta*(logits) - 1)** 2

     

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'bco_pair', 'sppo_hard', 'nca_pair', 'robust', 'exo_pair']"
            )

        chosen_frction_only_rewards = (
            self.beta
            * (
                policy_chosen_friction_logps.to(self.accelerator.device) - reference_chosen_friction_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_friction_only_rewards = (
            self.beta
            * (
                policy_rejected_friction_logps.to(self.accelerator.device)
                - reference_rejected_friction_logps.to(self.accelerator.device)
            ).detach()
        )



        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards, chosen_frction_only_rewards, rejected_friction_only_rewards
    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
        ) -> Tuple[torch.FloatTensor, ...]:  # Extended return type for friction variants
        """Run forward pass on concatenated chosen, rejected, and their friction variants.
        Returns logps and logits for both original and friction versions.
        """
        # print("batch keys in conc forward", batch.keys())
        
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=self.is_vision_model,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )

        # Get lengths for all variants
        len_chosen = batch["chosen_labels"].shape[0]
        len_rejected = batch["rejected_labels"].shape[0]
        len_chosen_friction = batch["chosen_friction_labels"].shape[0]
        len_rejected_friction = batch["rejected_friction_labels"].shape[0]

        # print("concatenated_batch keys", concatenated_batch.keys())
        # print("Lengths - chosen:", len_chosen, "rejected:", len_rejected,
        #         "chosen_friction:", len_chosen_friction, "rejected_friction:", len_rejected_friction)

        model_kwargs = {}
        if self.is_encoder_decoder:
            model_kwargs["labels"] = concatenated_batch["concatenated_labels"]
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

        if self.is_vision_model:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
            if "pixel_attention_mask" in concatenated_batch:
                model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Single forward pass for all variants
        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )
        all_logits = outputs.logits

        if all_logits.shape[:2] != concatenated_batch["concatenated_labels"].shape[:2]:
            # Handle llava case
            seq_len = concatenated_batch["concatenated_labels"].shape[1]
            all_logits = all_logits[:, -seq_len:]

        # Get log probabilities for all sequences
        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        # Calculate losses for both original and friction
        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])
        nll_loss_friction = cross_entropy_loss(
            all_logits[len_chosen + len_rejected:len_chosen + len_rejected + len_chosen_friction], 
            labels[len_chosen + len_rejected:len_chosen + len_rejected + len_chosen_friction]
        )

        if self.loss_type == "ipo" or self.loss_type == "friction" :
            all_logps = all_logps / size_completion

        # Split logps and logits for all variants
        # Original pairs
        start_idx = 0
        chosen_logps = all_logps[start_idx:start_idx + len_chosen]
        chosen_logits = all_logits[start_idx:start_idx + len_chosen]
        
        start_idx += len_chosen
        rejected_logps = all_logps[start_idx:start_idx + len_rejected]
        rejected_logits = all_logits[start_idx:start_idx + len_rejected]
        
        # Friction pairs
        start_idx += len_rejected
        chosen_friction_logps = all_logps[start_idx:start_idx + len_chosen_friction]
        chosen_friction_logits = all_logits[start_idx:start_idx + len_chosen_friction]
        
        start_idx += len_chosen_friction
        rejected_friction_logps = all_logps[start_idx:start_idx + len_rejected_friction]
        rejected_friction_logits = all_logits[start_idx:start_idx + len_rejected_friction]

        # print("Shapes after splitting:", 
        #         "\nChosen:", chosen_logps.shape, 
        #         "\nRejected:", rejected_logps.shape,
        #         "\nChosen Friction:", chosen_friction_logps.shape,
        #         "\nRejected Friction:", rejected_friction_logps.shape)

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits,
                    chosen_friction_logps, rejected_friction_logps, 
                    chosen_friction_logits, rejected_friction_logits,
                    nll_loss, nll_loss_friction, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, 
        nll_loss, chosen_friction_logps, rejected_friction_logps, 
        chosen_friction_logits, rejected_friction_logits,nll_loss_friction)
    
 

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        forward_output = self.concatenated_forward(model, batch)
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_nll_loss, 
       
            # Friction counterparts:
            policy_chosen_friction_logps,
            policy_rejected_friction_logps,
            policy_chosen_friction_logits,
            policy_rejected_friction_logits,
            policy_nll_loss_friction

        ) = forward_output 



        if self.aux_loss_enabled:
            aux_loss = forward_output[5]

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
            "reference_chosen_logps" in batch
            and "reference_rejected_logps" in batch
            and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        print("this ref_model =None is being run")
                        # ref_forward_output = self.concatenated_forward(self.model, batch)
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                            reference_chosen_friction_logps,
                            reference_rejected_friction_logps,
                            _,
                            _,
                            _,


                        ) = self.concatenated_forward(self.model, batch)
                    #    (
                    #         reference_chosen_logps,
                    #         reference_rejected_logps,
                    #         _,
                    #         _,
                    #         _,


                    #     ) = self.concatenated_forward(self.model, batch)
                else:
                    # print("this ref_model!=None is being run")
                    # ref_forward_output = self.concatenated_forward(self.ref_model, batch)
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                        reference_chosen_friction_logps,
                        reference_rejected_friction_logps,
                        _,
                        _,
                        _,

                    ) =self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards, chosen_friction_rewards, rejected_friction_rewards = self.friction_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            policy_chosen_friction_logps,
            policy_rejected_friction_logps,
            reference_chosen_friction_logps,
            reference_rejected_friction_logps)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_accuracies_friction = (chosen_friction_rewards > rejected_friction_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses * self.args.rpo_alpha + policy_nll_loss
            

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/chosen_fricton"] = chosen_friction_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected_friction"] = rejected_friction_rewards.mean().cpu()

        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}rewards/accuracies_friction"] = reward_accuracies_friction.mean().cpu()
        metrics[f"{prefix}rewards/margins_friction"] = (chosen_friction_rewards - rejected_friction_rewards).mean().cpu()

        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/rejected_friction"] = policy_rejected_friction_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen_friction"] = policy_chosen_friction_logps.detach().mean().cpu()


        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        metrics[f"{prefix}logits/rejected_friction"] = policy_rejected_friction_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen_friction"] = policy_chosen_friction_logits.detach().mean().cpu()

        metrics[f"{prefix}policy_nll_loss"] = policy_nll_loss.detach().mean().cpu()
        metrics[f"{prefix}policy_friction_nll_loss"] = policy_nll_loss_friction.detach().mean().cpu()
        metrics[f"{prefix}directrewards_student/accuracies"] = reward_accuracies.mean().cpu()




        if self.aux_loss_enabled:
            return losses.mean() + getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss, metrics

        return losses.mean(), metrics
    
 
    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.

        def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat(
                    [
                        tensor,
                        pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                    ],
                    dim=dim,
                )
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            print("get batch samples model", self.model)
            # print("batch", batch)
            print("batch keys", batch.keys())
            print("chosen chosen_input_ids in batch samples", batch['chosen_input_ids'])
            print("rejected rejected_input_ids in batch samples", batch['rejected_input_ids'])

            # Check if all elements in chosen_labels are the same
            chosen_labels_all_same = torch.all(batch['chosen_labels'] == batch['chosen_labels'][0]).item()
            print(f"All elements in chosen_labels are the same: {chosen_labels_all_same}")

            # Check if all elements in rejected_labels are the same
            rejected_labels_all_same = torch.all(batch['rejected_labels'] == batch['rejected_labels'][0]).item()
            print(f"All elements in rejected_labels are the same: {rejected_labels_all_same}")

            # Additionally, check if the two tensors are element-wise equal
            tensors_equal = torch.equal(batch['chosen_labels'], batch['rejected_labels'])
            print(f"chosen_labels and rejected_labels are exactly the same: {tensors_equal}")
            # print("batch", batch)
            print(f"Shape of 'prompt_input_ids': {batch['prompt_input_ids'].shape}")
            print(f"Shape of 'prompt_attention_mask': {batch['prompt_attention_mask'].shape}")
            print(f"Max length: {self.max_length}")
            print(f"Pad token ID: {self.tokenizer.pad_token_id}")

            # model is changed to self.model due to a generator object being shown here for the model
            policy_output = self.model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        # Assuming self.args.max_steps is 2000 or any other maximum step value

        quarter_steps = self.args.max_steps // 4
        batch_policy_token_count_list = []
        batch_chosen_token_count_list = []
        batch_rejected_token_count_list = []
        batch_reference_token_count_list = []
    

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval and (self.state.global_step % quarter_steps == 0 or self.state.global_step == self.args.max_steps):
            num_samples = len(dataloader.dataset)

            # Select a specific range of indices (e.g., first 20 samples)
            start_index = 0
            min_samples_for_generation_during_eval = 4
            end_index = min(min_samples_for_generation_during_eval, num_samples)
            selected_indices = list(range(start_index, end_index))

            # Use dataloader.dataset.select to get the specific batch without iterating over the DataLoader
            selected_dataset = dataloader.dataset.select(selected_indices)
            print("size of selected eval dataset for automatic metrics:", len(selected_dataset))
            batch_size = self.args.eval_batch_size

            # Initialize lists to hold metrics for averaging
            batch_rouge_scores_list = []
            batch_bleu_scores_list = []
            batch_meteor_scores_list = []
            batch_semantic_similarity_list = []

            batch_rouge_scores_list_ref = []
            batch_bleu_scores_list_ref = []
            batch_meteor_scores_list_ref = []
            batch_semantic_similarity_list_ref = []

            




            # Process the selected dataset in smaller batches
            for i in range(0, len(selected_dataset), batch_size):
                # Select a batch of samples
                
                batch_indices = list(range(i, min(i + batch_size, len(selected_dataset))))
                batch_dataset = selected_dataset.select(batch_indices)
                batch = self.data_collator(batch_dataset)
                batch = self._prepare_inputs(batch)
                # print("printing batch",batch.keys() )

                # Generate and process the batch
                print("getting batch generation inference", i)
                # policy_output_decoded = self.get_batch_samples(self.model, batch)
                policy_output_decoded, ref_output_decoded = self.get_batch_samples(self.model, batch)
                policy_output_decoded = [pol[len(prompt):] for prompt, pol in zip(batch['prompt'], policy_output_decoded)]
                ref_output_decoded = [ref[len(prompt):] for prompt, ref in zip(batch['prompt'], ref_output_decoded)]

                print("policy output",policy_output_decoded[0])
                # print("ref output",ref_output_decoded [0])
                # print("batch generation samples", policy_output_decoded)

                # Move tensors to CPU before converting to NumPy arrays
                chosen_labels = batch['chosen_labels'].cpu().numpy()
                rejected_labels = batch['rejected_labels'].cpu().numpy()
                # print("full keys batch",batch.keys() )

                # Process the labels for metrics computation
                chosen_labels = np.where(chosen_labels != -100, chosen_labels, self.tokenizer.pad_token_id)
                rejected_labels = np.where(rejected_labels != -100, rejected_labels, self.tokenizer.pad_token_id)
                chosen_output_decoded = self.tokenizer.batch_decode(chosen_labels, skip_special_tokens=True)
                rejected_output_decoded = self.tokenizer.batch_decode(rejected_labels, skip_special_tokens=True)
                print("chosen labels decoded",chosen_output_decoded )
                print("rejected labels decoded",chosen_output_decoded )
                policy_token_counts = [len(self.tokenizer.tokenize(pred)) for pred in policy_output_decoded]
                reference_token_counts = [len(self.tokenizer.tokenize(pred)) for pred in ref_output_decoded]
                chosen_token_counts = [len(self.tokenizer.tokenize(chosen)) for chosen in chosen_output_decoded]
                rejected_token_counts = [len(self.tokenizer.tokenize(rejected)) for rejected in rejected_output_decoded]

                # Append token counts to batch lists
                batch_policy_token_count_list.append(np.mean(policy_token_counts))
                batch_reference_token_count_list.append(np.mean(reference_token_counts))
                batch_chosen_token_count_list.append(np.mean(chosen_token_counts))
                batch_rejected_token_count_list.append(np.mean(rejected_token_counts))

                                # ROUGE expects a newline after each sentence
                decoded_preds = ["\n".join(pred.strip()) for pred in policy_output_decoded]
                decoded_ref_preds = ["\n".join(pred.strip()) for pred in ref_output_decoded]
                decoded_chosen_labels = ["\n".join(label.strip()) for label in chosen_output_decoded]

                # Compute ROUGE scores using RougeScorer for policy predictions
                rouge_scores = self.scorer.score("\n".join(decoded_chosen_labels), "\n".join(decoded_preds))
                rouge_scores = {k: v.fmeasure * 100 for k, v in rouge_scores.items()}  # Extract F1 scores
                rouge_scores = {k: round(v, 4) for k, v in rouge_scores.items()}

                # Compute ROUGE scores for reference model predictions
                rouge_scores_ref = self.scorer.score("\n".join(decoded_chosen_labels), "\n".join(decoded_ref_preds))
                rouge_scores_ref = {k: v.fmeasure * 100 for k, v in rouge_scores_ref.items()}  # Extract F1 scores
                rouge_scores_ref = {k: round(v, 4) for k, v in rouge_scores_ref.items()}

                # Append ROUGE scores for the batch
                batch_rouge_scores_list.append(rouge_scores)
                batch_rouge_scores_list_ref.append(rouge_scores_ref)

                # Compute BLEU score
                                # Compute BLEU score for policy predictions
                                # Compute BLEU score for policy model predictions
                self.bleu_metric.add_batch(
                    predictions=[pred.split() for pred in decoded_preds],
                    references=[[label.split()] for label in decoded_chosen_labels]
                )
                batch_bleu_scores_list.append(self.bleu_metric.compute())

                # Directly compute BLEU score for reference model predictions
                self.bleu_metric.add_batch(
                    predictions=[pred.split() for pred in decoded_ref_preds],
                    references=[[label.split()] for label in decoded_chosen_labels]
                )
                batch_bleu_scores_list_ref.append(self.bleu_metric.compute())



                

                # Compute METEOR score
                                # Compute METEOR score for policy predictions
                                # Compute METEOR score for policy model predictions
                for pred, ref in zip(decoded_preds, decoded_chosen_labels):
                    self.meteor_metric.add(prediction=pred, reference=ref)
                batch_meteor_scores_list.append(self.meteor_metric.compute())

                # Directly compute METEOR score for reference model predictions
                for pred, ref in zip(decoded_ref_preds, decoded_chosen_labels):
                    self.meteor_metric.add(prediction=pred, reference=ref)
                batch_meteor_scores_list_ref.append(self.meteor_metric.compute())


                 # Compute semantic similarity for policy predictions
                embeddings1 = self.sentence_transformer_model.encode(decoded_preds, convert_to_tensor=True)
                embeddings2 = self.sentence_transformer_model.encode(decoded_chosen_labels, convert_to_tensor=True)
                semantic_similarities = util.pytorch_cos_sim(embeddings1, embeddings2)
                avg_similarity = semantic_similarities.mean().item()

                # Append semantic similarity for policy predictions
                batch_semantic_similarity_list.append(avg_similarity)

                # Compute semantic similarity for reference model predictions
                embeddings1_ref = self.sentence_transformer_model.encode(decoded_ref_preds, convert_to_tensor=True)
                semantic_similarities_ref = util.pytorch_cos_sim(embeddings1_ref, embeddings2)
                avg_similarity_ref = semantic_similarities_ref.mean().item()

                # Append semantic similarity for reference model predictions
                batch_semantic_similarity_list_ref.append(avg_similarity_ref)


                # Step 1: Create the DataFrame with the decoded outputs
                df_log = pd.DataFrame({
                    "Prompt": batch["prompt"],
                    "Policy": [pol for prompt, pol in zip(batch["prompt"], policy_output_decoded)],
                    "Reference": [pol for prompt, pol in zip(batch["prompt"], ref_output_decoded)],
                    "Chosen Output": chosen_output_decoded,
                    "Rejected Output": rejected_output_decoded
                })

         
                log_dict = df_log.to_dict('list')

                # Log each column of the DataFrame using `self.log`
                for column, values in log_dict.items():
                    self.log({f"{column}_batch_{i//batch_size + 1}": values})

   
            avg_policy_tokens = round(np.mean(batch_policy_token_count_list), 2)
            avg_reference_tokens = round(np.mean(batch_reference_token_count_list), 2)
            avg_chosen_tokens = round(np.mean(batch_chosen_token_count_list), 2)
            avg_rejected_tokens = round(np.mean(batch_rejected_token_count_list), 2)

 

            # Log the average token counts
            self.log({
                "avg_policy_tokens": avg_policy_tokens,
                "avg_reference_tokens":avg_reference_tokens,
                "avg_chosen_tokens": avg_chosen_tokens,
                "avg_rejected_tokens": avg_rejected_tokens
            })
                        # Compute the average ROUGE scores across all batches (for policy and reference)
                        # Compute the average ROUGE scores across all batches (for policy and reference)
            if batch_rouge_scores_list:
                overall_rouge_scores = {
                    key: round(sum([batch_scores[key] for batch_scores in batch_rouge_scores_list]) / len(batch_rouge_scores_list), 2)
                    for key in batch_rouge_scores_list[0].keys()
                }
                # print("Overall ROUGE scores (policy):", overall_rouge_scores)
                self.log({"overall_rouge_scores": overall_rouge_scores})

            if batch_rouge_scores_list_ref:
                overall_rouge_scores_ref = {
                    key: round(sum([batch_scores[key] for batch_scores in batch_rouge_scores_list_ref]) / len(batch_rouge_scores_list_ref), 2)
                    for key in batch_rouge_scores_list_ref[0].keys()
                }
                # print("Overall ROUGE scores (reference):", overall_rouge_scores_ref)
                self.log({"overall_rouge_scores_ref": overall_rouge_scores_ref})

            # Compute the average BLEU scores across all batches (for policy and reference)
            if batch_bleu_scores_list:
                overall_bleu_scores = {
                    key: round(sum([batch_scores[key] if isinstance(batch_scores[key], (int, float)) else np.mean(batch_scores[key])
                                    for batch_scores in batch_bleu_scores_list]) / len(batch_bleu_scores_list), 2)
                    for key in batch_bleu_scores_list[0].keys()
                }
                # print("Overall BLEU scores (policy):", overall_bleu_scores)
                self.log({"overall_bleu_scores": overall_bleu_scores})

            if batch_bleu_scores_list_ref:
                overall_bleu_scores_ref = {
                    key: round(sum([batch_scores[key] if isinstance(batch_scores[key], (int, float)) else np.mean(batch_scores[key])
                                    for batch_scores in batch_bleu_scores_list_ref]) / len(batch_bleu_scores_list_ref), 2)
                    for key in batch_bleu_scores_list_ref[0].keys()
                }
                # print("Overall BLEU scores (reference):", overall_bleu_scores_ref)
                self.log({"overall_bleu_scores_ref": overall_bleu_scores_ref})

            # Compute the average METEOR scores across all batches (for policy and reference)
            if batch_meteor_scores_list:
                overall_meteor_scores = {
                    key: round(sum([batch_scores[key] for batch_scores in batch_meteor_scores_list]) / len(batch_meteor_scores_list), 2)
                    for key in batch_meteor_scores_list[0].keys()
                }
                # print("Overall POLICY METEOR scores:", overall_meteor_scores)
                self.log({"overall_meteor_scores": overall_meteor_scores})

            if batch_meteor_scores_list_ref:
                overall_meteor_scores_ref = {
                    key: round(sum([batch_scores[key] for batch_scores in batch_meteor_scores_list_ref]) / len(batch_meteor_scores_list_ref), 2)
                    for key in batch_meteor_scores_list_ref[0].keys()
                }
                # print("Overall REF METEOR scores:", overall_meteor_scores_ref)
                self.log({"overall_meteor_scores_ref": overall_meteor_scores_ref})

            # Compute the average semantic similarity across all batches (for policy and reference)
            if batch_semantic_similarity_list:
                overall_semantic_similarity = round(sum(batch_semantic_similarity_list) / len(batch_semantic_similarity_list), 2)
                # print("Overall Semantic Similarity (policy):", overall_semantic_similarity)
                self.log({"overall_semantic_similarity": overall_semantic_similarity})

            if batch_semantic_similarity_list_ref:
                overall_semantic_similarity_ref = round(sum(batch_semantic_similarity_list_ref) / len(batch_semantic_similarity_list_ref), 2)
                # print("Overall Semantic Similarity (reference):", overall_semantic_similarity_ref)
                self.log({"overall_semantic_similarity_ref": overall_semantic_similarity_ref})



        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

