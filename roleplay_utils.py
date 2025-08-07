

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
from datasets import Dataset,load_dataset, DatasetDict
from datasets import load_from_disk
import re
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from tqdm import tqdm
# from datasets import load_metric


def compute_metrics(data):
    """
    Compute metrics for all dialogues.
    
    Parameters:
    - data: Dictionary containing dialogue data
    
    Returns:
    - Dictionary with computed metrics
    """
    metrics = {
        'quality': {},
        'efficiency': {},
        'persuasiveness': {},
        'friction_scores': {},  # Add new friction scores category
        'dialogues': [],  # Store detailed metrics for each dialogue
    }
    
    # Process each dialogue
    for dialogue_id, dialogue_data in tqdm(data.items(), desc="Processing dialogues"):
        dialogue_metrics = compute_dialogue_metrics(dialogue_data)
        metrics['dialogues'].append(dialogue_metrics)
       
    
    # Compute average metrics across all dialogues
    metrics['quality'] = compute_average_quality_metrics(metrics['dialogues'])
    metrics['efficiency'] = compute_average_efficiency_metrics(metrics['dialogues'])
    metrics['persuasiveness'] = compute_average_persuasiveness_metrics(metrics['dialogues'])
    metrics['friction_scores'] = compute_average_friction_scores(metrics['dialogues'])
    
    return metrics

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

def compute_dialogue_metrics(dialogue_data):
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
        'friction_scores':[]
    }
    
    # Track resolved blocks
    resolved_blocks = set()
    prev_resolved_blocks = set()
    print("len of dialogues", len(dialogue_data['turns']))
    # Process each turn
    for i, turn_data in enumerate(dialogue_data['turns']):

        # add friction scores for each turn and then compute mean, std in compute_average_friction_scores
        # if 'parsed_gpt_response' in turn_data:
        dialogue_metrics['friction_scores'].append(turn_data['parsed_gpt_response']['friction_score'])
            # print(turn_data['parsed_gpt_response']['friction_score'])
        
        turn_metrics = compute_turn_metrics(turn_data, i, resolved_blocks, prev_resolved_blocks)
        dialogue_metrics['turn_metrics'].append(turn_metrics)
        
        # Update resolved blocks tracking
        if 'parsed_gpt_response' in turn_data and 'resolved_blocks' in turn_data['parsed_gpt_response']:
            curr_resolved = extract_resolved_blocks(turn_data['parsed_gpt_response']['resolved_blocks'])
            prev_resolved_blocks = resolved_blocks.copy()
            resolved_blocks = resolved_blocks.union(curr_resolved)
        
        # Store current resolved blocks count and blocks resolved in this turn
        dialogue_metrics['resolved_blocks_count'].append(len(resolved_blocks))
        blocks_in_turn = len(resolved_blocks) - len(prev_resolved_blocks)
        dialogue_metrics['blocks_resolved_per_turn'].append(blocks_in_turn)
    
    # Compute dialogue-level metrics
    dialogue_metrics['final_blocks_resolved'] = len(resolved_blocks)
    dialogue_metrics['resolution_rate'] = len(resolved_blocks) / len(STANDARD_BLOCKS)
    
    # List the actual resolved blocks
    dialogue_metrics['resolved_blocks'] = list(resolved_blocks)
    
    # Check if all blocks were resolved
    all_blocks_resolved = len(resolved_blocks) == len(STANDARD_BLOCKS)
    dialogue_metrics['all_blocks_resolved'] = all_blocks_resolved
    
    # Calculate turns until resolution if all blocks were resolved
    if all_blocks_resolved:
        for i, count in enumerate(dialogue_metrics['resolved_blocks_count']):
            if count == len(STANDARD_BLOCKS):
                dialogue_metrics['turns_until_resolution'] = i + 1
                break
    else:
        dialogue_metrics['turns_until_resolution'] = dialogue_metrics['total_turns']
    
    return dialogue_metrics

def compute_turn_metrics(turn_data, turn_index, resolved_blocks, prev_resolved_blocks):
    """
    Compute metrics for a single turn.
    
    Parameters:
    - turn_data: Dictionary containing data for a single turn
    - turn_index: Index of the turn
    - resolved_blocks: Set of blocks resolved so far
    - prev_resolved_blocks: Set of blocks resolved before this turn
    
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
    
    # Quality metrics - Semantic similarity between friction agent and GPT friction
    if 'parsed_friction' in turn_data and 'parsed_gpt_response' in turn_data and 'friction_statement' in turn_data['parsed_gpt_response'] and turn_data['parsed_gpt_response']['friction_statement']:
        agent_friction = turn_data['parsed_friction']
        gpt_friction = turn_data['parsed_gpt_response']['friction_statement']
        
        # ROUGE scores
        rouge_scores = rouge_scorer.score(agent_friction, gpt_friction)
        turn_metrics['quality']['rouge1_f'] = rouge_scores['rouge1'].fmeasure
        turn_metrics['quality']['rouge2_f'] = rouge_scores['rouge2'].fmeasure
        turn_metrics['quality']['rougeL_f'] = rouge_scores['rougeL'].fmeasure
        
        # BLEU score
        agent_tokens = word_tokenize(agent_friction)
        gpt_tokens = word_tokenize(gpt_friction)
        smoother = SmoothingFunction().method1
        if agent_tokens and gpt_tokens:
            turn_metrics['quality']['bleu'] = corpus_bleu([[gpt_tokens]], [agent_tokens], smoothing_function=smoother)
        else:
            turn_metrics['quality']['bleu'] = 0
        
        # Semantic similarity
        agent_embedding = semantic_model.encode(agent_friction, convert_to_tensor=True)
        gpt_embedding = semantic_model.encode(gpt_friction, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(agent_embedding, gpt_embedding).item()
        turn_metrics['quality']['semantic_similarity'] = similarity
        
        # NLI analysis - check entailment between agent friction and GPT friction
        if len(agent_friction) > 5 and len(gpt_friction) > 5:  # Ensure non-empty strings
            # Agent -> GPT direction
            result_agent_to_gpt = nli_model(agent_friction, [gpt_friction], hypothesis_template="{}.")
            turn_metrics['quality']['agent_to_gpt_entailment'] = result_agent_to_gpt['scores'][0]
            
            # GPT -> Agent direction
            result_gpt_to_agent = nli_model(gpt_friction, [agent_friction], hypothesis_template="{}.")
            turn_metrics['quality']['gpt_to_agent_entailment'] = result_gpt_to_agent['scores'][0]
    
    # Persuasiveness metrics - Check if the friction was adopted
    if 'parsed_friction' in turn_data and 'gpt_utterances' in turn_data and turn_data['gpt_utterances']:
        agent_friction = turn_data['parsed_friction']
        gpt_utterances = ' '.join(turn_data['gpt_utterances'])
        
        # Semantic similarity between friction and utterances to check adoption
        agent_embedding = semantic_model.encode(agent_friction, convert_to_tensor=True)
        utterances_embedding = semantic_model.encode(gpt_utterances, convert_to_tensor=True)
        adoption_similarity = util.pytorch_cos_sim(agent_embedding, utterances_embedding).item()
        turn_metrics['persuasiveness']['adoption_similarity'] = adoption_similarity
        
        # Simple heuristic for adoption: consider adopted if similarity > 0.5
        turn_metrics['persuasiveness']['adopted'] = adoption_similarity > 0.5
        
        # Check for acknowledgment without adoption
        # Look for acknowledgment phrases followed by counterpoints
        acknowledgment_patterns = [
            r"(?i)that'?s? (?:a good|an interesting|a valid) point.+but",
            r"(?i)i (?:understand|see|get) (?:your|that) (?:point|concern).+but",
            r"(?i)you'?re? right.+however",
            r"(?i)that'?s? true.+(?:although|though)"
        ]
        acknowledged_without_adoption = any(re.search(pattern, gpt_utterances) for pattern in acknowledgment_patterns)
        turn_metrics['persuasiveness']['acknowledged_without_adoption'] = acknowledged_without_adoption
    
    # Efficiency metrics - Blocks resolved in this turn
    turn_metrics['blocks_resolved'] = len(resolved_blocks) - len(prev_resolved_blocks)
    turn_metrics['total_blocks_resolved'] = len(resolved_blocks)
    turn_metrics['resolved_blocks'] = list(resolved_blocks)
    
    # Store utterances for analysis
    turn_metrics['gpt_utterances'] = turn_data.get('gpt_utterances', [])
    
    return turn_metrics

def compute_average_friction_scores(all_dialogues):
    """
    Compute average friction scores and their standard deviations across all dialogues.
    
    Args:
        all_dialogues (list): List of dialogue metrics
        
    Returns:
        dict: Statistics about friction scores
    """
    # Create a list to collect all friction scores
    friction_scores = []
    
    for dialogue in all_dialogues:
        # friction_scores = dialogue['friction_scores']
        friction_scores.append(dialogue['friction_scores'])
        print(friction_scores, type(friction_scores))
        # for turn in dialogue['turn_metrics']:
        #     print("turn", turn)
       
        #     # Look for friction_score directly in parsed_gpt_response
        #     if 'parsed_gpt_response' in turn and 'friction_score' in turn['parsed_gpt_response']:
        #         print(turn['parsed_gpt_response'])
        #         score = turn['parsed_gpt_response']['friction_score']
        #         if score is not None:
        #             friction_scores.append(score)
    
    # Compute statistics (mean and std)
    stats = {}

    # friction_scores = [item for sublist in friction_scores for item in sublist]
    friction_scores = [item for sublist in friction_scores for item in sublist if item is not None]
    if friction_scores:
        # Using numpy for more accurate statistics
        import numpy as np
        values_array = np.array(friction_scores)
        stats = {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'count': len(friction_scores)
        }
    else:
        stats = {
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'count': 0
        }
    
    return stats

def compute_average_quality_metrics(all_dialogues):
    """
    Compute average quality metrics and their standard deviations across all dialogues.
    """
    quality_metrics = {
        'semantic_similarity': [],
        'rouge1_f': [],
        'rouge2_f': [],
        'rougeL_f': [],
        'bleu': [],
        'agent_to_gpt_entailment': [],
        'gpt_to_agent_entailment': []
    }
    
    # Add bidirectional entailment field
    quality_metrics['bidirectional_entailment'] = []
    
    for dialogue in all_dialogues:
        for turn in dialogue['turn_metrics']:
            if 'quality' in turn and turn['quality']:
                for metric, value in turn['quality'].items():
                    if metric in quality_metrics and value is not None:
                        quality_metrics[metric].append(value)
                
                # Calculate and store bidirectional entailment if both metrics exist
                if 'agent_to_gpt_entailment' in turn['quality'] and 'gpt_to_agent_entailment' in turn['quality']:
                    agent_to_gpt = turn['quality']['agent_to_gpt_entailment']
                    gpt_to_agent = turn['quality']['gpt_to_agent_entailment']
                    if agent_to_gpt is not None and gpt_to_agent is not None:
                        bidirectional = (agent_to_gpt + gpt_to_agent) / 2
                        quality_metrics['bidirectional_entailment'].append(bidirectional)
    
    # Compute statistics (mean and std)
    stats_metrics = {}
    for metric, values in quality_metrics.items():
        if values:
            # Using numpy for more accurate statistics
            import numpy as np
            values_array = np.array(values)
            stats_metrics[metric] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'count': len(values)
            }
        else:
            stats_metrics[metric] = {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'count': 0
            }
    
    return stats_metrics


def compute_average_efficiency_metrics(all_dialogues):
   """
   Compute average efficiency metrics and their standard deviations across all dialogues.
   """
   efficiency_metrics = {
       'total_turns': [],
       'turns_until_resolution': [],
       'resolution_rate': [],
       'blocks_resolved_per_turn': []
   }
   
   for dialogue in all_dialogues:
       efficiency_metrics['total_turns'].append(dialogue['total_turns'])
       efficiency_metrics['turns_until_resolution'].append(dialogue['turns_until_resolution'])
       efficiency_metrics['resolution_rate'].append(dialogue['resolution_rate'])
       
       # Average blocks resolved per turn for this dialogue
       avg_blocks_per_turn = sum(dialogue['blocks_resolved_per_turn']) / dialogue['total_turns']
       efficiency_metrics['blocks_resolved_per_turn'].append(avg_blocks_per_turn)
   
   # Compute statistics (mean and std)
   stats_metrics = {}
   for metric, values in efficiency_metrics.items():
       if values:
           # Using numpy for more accurate statistics
           import numpy as np
           values_array = np.array(values)
           stats_metrics[metric] = {
               'mean': float(np.mean(values_array)),
               'std': float(np.std(values_array)),
               'min': float(np.min(values_array)),
               'max': float(np.max(values_array)),
               'count': len(values)
           }
       else:
           stats_metrics[metric] = {
               'mean': None,
               'std': None,
               'min': None,
               'max': None,
               'count': 0
           }
   
   return stats_metrics

def compute_average_persuasiveness_metrics(all_dialogues):
   """
   Compute average persuasiveness metrics and their standard deviations across all dialogues.
   """
   persuasiveness_metrics = {
       'adoption_rate': [],
       'acknowledgment_without_adoption_rate': [],
       'adoption_similarity': []
   }
   
   for dialogue in all_dialogues:
       dialogue_adoptions = []
       dialogue_acknowledgments = []
       dialogue_similarities = []
       
       for turn in dialogue['turn_metrics']:
           if 'persuasiveness' in turn and turn['persuasiveness']:
               if 'adopted' in turn['persuasiveness']:
                   dialogue_adoptions.append(turn['persuasiveness']['adopted'])
               if 'acknowledged_without_adoption' in turn['persuasiveness']:
                   dialogue_acknowledgments.append(turn['persuasiveness']['acknowledged_without_adoption'])
               if 'adoption_similarity' in turn['persuasiveness']:
                   dialogue_similarities.append(turn['persuasiveness']['adoption_similarity'])
       
       # Calculate rates for this dialogue
       if dialogue_adoptions:
           persuasiveness_metrics['adoption_rate'].append(sum(dialogue_adoptions) / len(dialogue_adoptions))
       if dialogue_acknowledgments:
           persuasiveness_metrics['acknowledgment_without_adoption_rate'].append(sum(dialogue_acknowledgments) / len(dialogue_acknowledgments))
       if dialogue_similarities:
           persuasiveness_metrics['adoption_similarity'].append(sum(dialogue_similarities) / len(dialogue_similarities))
   
   # Compute statistics (mean and std)
   stats_metrics = {}
   for metric, values in persuasiveness_metrics.items():
       if values:
           # Using numpy for more accurate statistics
           import numpy as np
           values_array = np.array(values)
           stats_metrics[metric] = {
               'mean': float(np.mean(values_array)),
               'std': float(np.std(values_array)),
               'min': float(np.min(values_array)),
               'max': float(np.max(values_array)),
               'count': len(values)
           }
       else:
           stats_metrics[metric] = {
               'mean': None,
               'std': None,
               'min': None,
               'max': None,
               'count': 0
           }
   
   return stats_metrics

def get_dialogue_rankings(metrics):
    """
    Rank dialogues by various metrics for analysis.
    
    Parameters:
    - metrics: Dictionary containing computed metrics
    
    Returns:
    - Dictionary with rankings
    """
    dialogues = metrics['dialogues']
    rankings = {
        'fastest_resolution': [],  # Dialogues that resolved all blocks in fewest turns
        'highest_adoption_rate': [],  # Dialogues with highest friction adoption rate
        'highest_semantic_similarity': [],  # Dialogues with highest agent-GPT friction similarity
        'most_blocks_resolved': [],  # Dialogues that resolved the most blocks
        'lowest_quality': [],  # Dialogues with lowest quality metrics
    }
    
    # Calculate dialogue-level average metrics for ranking
    for dialogue in dialogues:
        # Calculate average semantic similarity for this dialogue
        avg_similarity = 0
        similarity_count = 0
        adoption_count = 0
        adoption_total = 0
        
        for turn in dialogue['turn_metrics']:
            if 'quality' in turn and 'semantic_similarity' in turn['quality']:
                avg_similarity += turn['quality']['semantic_similarity']
                similarity_count += 1
            
            if 'persuasiveness' in turn and 'adopted' in turn['persuasiveness']:
                adoption_total += 1
                if turn['persuasiveness']['adopted']:
                    adoption_count += 1
        
        dialogue['avg_semantic_similarity'] = avg_similarity / similarity_count if similarity_count > 0 else 0
        dialogue['adoption_rate'] = adoption_count / adoption_total if adoption_total > 0 else 0
    
    # Rank by resolution speed (turns until resolution)
    resolution_sorted = sorted([d for d in dialogues if d.get('all_blocks_resolved', False)], 
                              key=lambda x: x['turns_until_resolution'])
    rankings['fastest_resolution'] = [{'dialogue_id': d['dialogue_id'], 
                                     'turns_until_resolution': d['turns_until_resolution']} 
                                    for d in resolution_sorted[:5]]
    
    # Rank by adoption rate
    adoption_sorted = sorted(dialogues, key=lambda x: x.get('adoption_rate', 0), reverse=True)
    rankings['highest_adoption_rate'] = [{'dialogue_id': d['dialogue_id'], 
                                         'adoption_rate': d.get('adoption_rate', 0)} 
                                        for d in adoption_sorted[:5]]
    
    # Rank by semantic similarity
    similarity_sorted = sorted(dialogues, key=lambda x: x.get('avg_semantic_similarity', 0), reverse=True)
    rankings['highest_semantic_similarity'] = [{'dialogue_id': d['dialogue_id'], 
                                              'avg_semantic_similarity': d.get('avg_semantic_similarity', 0)} 
                                             for d in similarity_sorted[:5]]
    
    # Rank by blocks resolved
    resolved_sorted = sorted(dialogues, key=lambda x: x.get('final_blocks_resolved', 0), reverse=True)
    rankings['most_blocks_resolved'] = [{'dialogue_id': d['dialogue_id'], 
                                        'blocks_resolved': d.get('final_blocks_resolved', 0)} 
                                       for d in resolved_sorted[:5]]
    
    # Rank by lowest quality (for identifying problematic dialogues)
    quality_sorted = sorted(dialogues, key=lambda x: x.get('avg_semantic_similarity', 0))
    rankings['lowest_quality'] = [{'dialogue_id': d['dialogue_id'], 
                                 'avg_semantic_similarity': d.get('avg_semantic_similarity', 0)} 
                                for d in quality_sorted[:5]]
    
    return rankings

def identify_difficult_blocks(metrics):
    """
    Identify which blocks tend to be more difficult to resolve.
    
    Parameters:
    - metrics: Dictionary containing computed metrics
    
    Returns:
    - Dictionary with block difficulty analysis
    """
    block_metrics = {block: {'resolved_count': 0, 'avg_turn_resolved': []} for block in STANDARD_BLOCKS}
    total_dialogues = len(metrics['dialogues'])
    
    for dialogue in metrics['dialogues']:
        # Track when each block was resolved
        resolved_blocks_by_turn = {}
        
        for turn_idx, turn in enumerate(dialogue['turn_metrics']):
            if 'resolved_blocks' in turn:
                for block in turn['resolved_blocks']:
                    if block not in resolved_blocks_by_turn and block in STANDARD_BLOCKS:
                        resolved_blocks_by_turn[block] = turn_idx + 1
        
        # Update block metrics
        for block in STANDARD_BLOCKS:
            if block in resolved_blocks_by_turn:
                block_metrics[block]['resolved_count'] += 1
                block_metrics[block]['avg_turn_resolved'].append(resolved_blocks_by_turn[block])
    
    # Calculate averages and resolution rates
    for block, data in block_metrics.items():
        data['resolution_rate'] = data['resolved_count'] / total_dialogues
        data['avg_turn_to_resolve'] = sum(data['avg_turn_resolved']) / len(data['avg_turn_resolved']) if data['avg_turn_resolved'] else None
    
    # Rank blocks by difficulty (lower resolution rate = more difficult)
    block_difficulty = sorted(STANDARD_BLOCKS, key=lambda x: block_metrics[x]['resolution_rate'])
    
    return {
        'block_metrics': block_metrics,
        'difficulty_ranking': block_difficulty
    }
