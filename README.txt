# FRICTION AGENT++ Implementation Files

## Main Files
- **`friction_config.py`**  
  Configuration and hyperparameter settings for FRICTION AGENT++ training.  
  - Defines FRICTION AGENT++ and ablation losses.

- **`friction_plus_plus_training`**  
  Main training script for FRICTION++ AGENT.  
  - Supports DPO (`loss_type="sigmoid"`) and IPO (`loss_type="ipo"`) baselines.  
  - Manages hyperparameter sweeps and logging.

- **`friction_trainer.py`**  
  Core implementation of FRICTION++ loss with TRL integration.  
  - Computes loss and handles phi-unconditioned forward passes.  
  - Implements preference alignment.

- **`friction_roleplay_collaboration.py`**  
  Roleplaying loop for data generation.  
  - Records data for counterfactual evaluation.  
  - Produces turn-based dialogue between friction and collaborator agents.

- **`RM.py`**  
  Reward modeling implementation.  
  - Utilizes OPT models for reward computation.  
  - Supports PPO training.

- **`bc_expert.py`**  
  BC-EXPERT baseline implementation.  
  - Trains a base instruct model with SFT on all friction tokens (not just trajectory-end completions).  
  - Excludes loss computation on non-completion state parts.

- **`counterfactual_eval.py`**  
  Counterfactual evaluation of baselines on SFT trajectories.  
  - Uses data sampled from `friction_roleplay_collaboration.py`.  
  - Computes rewards, margins, and win rates against the SFT model.  
  - Generates evaluation metrics.

- **`common_ground.py`**  
  Computes common ground metrics for WTD and DELI datasets.

- **`roleplay_utils.py`**  
  Helper functions for parsing blocks, cards, and computing basic metrics.
 