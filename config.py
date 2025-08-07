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


@dataclass
class FAAFConfig(TrainingArguments):
    r"""
    Initialize DROConfig.
    friction Configuration is customized from the original DPO Config file in the TRL Library
    Args:
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper.
        label_smoothing (`float`, defaults to 0):
            The robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report and [Robust DPO](https://huggingface.co/papers/2403.00409) paper that should be between 0 and 0.5.
        loss_type (`str`, defaults to `"sigmoid"`):
            The type of friction loss to use. 
            faaf_not_conditioned: friction objective with no phi (frictive state) conditioning
            faaf_first_part_only: friction objective with only the first phi-conditioned part
            friction: Complete friction objective with both terms
             
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`Optional[int]`, *optional*):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Flag to precompute reference model log probabilities for training and evaluation datasets. This is useful if you want to train
            without the reference model and reduce the total GPU memory needed.
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        ref_model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the ref model from a string
        model_adapter_name (`str`, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        reference_free (`bool` defaults to `False`):
            If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.
        force_use_ref_model (`bool`, defaults to `False`):
            In case one passes a PEFT model for the active model and you want to use a different model for the ref_model, set this flag to `True`.
        f_divergence_type (`FDivergenceType`, *optional*, defaults to `FDivergenceType.REVERSE_KL`):
            The type of f-divergence regularization function to compute divergence between policy and reference model. This argument is optional, defaults to `FDivergenceType.REVERSE_KL`.
        f_alpha_divergence_coef (`float`, *optional*, defaults to `1.0`):
            The alpha coef in alpha-divergence(u^-alpha) regularization function for DPO loss.
        sync_ref_model ('bool', defaults to `False`):
            The flag for syncing reference model during training from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha ('float', defaults to 1.0):
            The alpha parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_sync_steps ('int', defaults to 2):
            The tau parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        rpo_alpha ('float', defaults to `None`):
            The alpha parameter from the [RPO](https://huggingface.co/papers/2404.19733) paper. If None, no weighting is applied and the loss is the same as the DPO loss.
    """

    beta: float = 0.1
    label_smoothing: float = 0
    loss_type: Literal["sigmoid", "ipo","friction",  "friction_not_conditioned", "friction_first_part_only"] = "friction",
   
    label_pad_token_id: int = -100
    padding_value: Optional[int] = None
    truncation_mode: str = "keep_end"
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_target_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = True
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    f_divergence_type: Optional[FDivergenceType] = FDivergenceType.REVERSE_KL
    f_alpha_divergence_coef: Optional[float] = 1.0
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    rpo_alpha: Optional[float] = None

    def __post_init__(self):
        if self.loss_type == "kto_pair":
            raise ValueError("Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.")
        return super().__post_init__()

