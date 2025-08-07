
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import pandas as pd
import os
from matplotlib.ticker import MaxNLocator

def compute_common_ground_metrics(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute metrics for common ground size and growth for a single model.
    """
    model_metrics = {}
    
    for dialogue_id, dialogue_data in model_data.items():
        dialogue_metrics = {
            "turn_metrics": [],
            "cumulative_metrics": [],
            "growth_metrics": []
        }
        
        # Store cumulative common ground propositions
        cumulative_common_ground = {
            "equality": {},
            "inequality": {},
            "order": {}
        }
        
        prev_turn_count = {"equality": 0, "inequality": 0, "order": 0, "total": 0}
        
        for turn_data in dialogue_data.get("turns", []):
            turn_idx = turn_data.get("turn_idx")
            common_ground = turn_data.get("parsed_common_ground", {})
            
            # Get counts for this turn
            turn_counts = count_common_ground_propositions(common_ground)
            
            # Update cumulative common ground
            # For equality and inequality (similar structure)
            for category in ["equality", "inequality"]:
                for block, relations in common_ground.get(category, {}).items():
                    if block not in cumulative_common_ground[category]:
                        cumulative_common_ground[category][block] = []
                    
                    # Add new relations that aren't already in cumulative
                    for relation in relations:
                        if relation not in cumulative_common_ground[category][block]:
                            cumulative_common_ground[category][block].append(relation)
            
            # For order (more complex structure)
            # For order (more complex structure)
            for block, relations in common_ground.get("order", {}).items():
                if block not in cumulative_common_ground["order"]:
                    cumulative_common_ground["order"][block] = {">": [], "<": []}
                
                # Check if relations is a dictionary with direction keys
                if isinstance(relations, dict):
                    for direction in [">", "<"]:
                        if direction in relations:
                            # Get the relations list (handle both list and non-list cases)
                            rel_list = relations[direction] if isinstance(relations[direction], list) else [relations[direction]]
                            for relation in rel_list:
                                if relation not in cumulative_common_ground["order"][block][direction]:
                                    cumulative_common_ground["order"][block][direction].append(relation)
                # Handle case where relations is a list
                elif isinstance(relations, list):
                    # Default to ">" direction if not specified
                    for relation in relations:
                        if relation not in cumulative_common_ground["order"][block][">"]:
                            cumulative_common_ground["order"][block][">"].append(relation)
                # Handle case where relations is something else
                else:
                    # Default to ">" direction if not specified
                    if relations not in cumulative_common_ground["order"][block][">"]:
                        cumulative_common_ground["order"][block][">"].append(relations)

                        # Get counts for cumulative common ground
            cumulative_counts = count_common_ground_propositions(cumulative_common_ground)
            max_possible_propositions = 37 ##calculate_theoretical_maximum()   
            # Add normalized metrics
            normalized_counts = {
                category: count / max_possible_propositions 
                for category, count in cumulative_counts.items()
            }


            # Calculate growth from previous turn
            growth = {
                "equality": cumulative_counts["equality"] - prev_turn_count["equality"],
                "inequality": cumulative_counts["inequality"] - prev_turn_count["inequality"],
                "order": cumulative_counts["order"] - prev_turn_count["order"],
                "total": cumulative_counts["total"] - prev_turn_count["total"]
            }
            
            # Store metrics for this turn
            dialogue_metrics["turn_metrics"].append({
                "turn_idx": turn_idx,
                "counts": turn_counts
            })
            
            dialogue_metrics["cumulative_metrics"].append({
                "turn_idx": turn_idx,
                "counts": cumulative_counts
            })
            
            dialogue_metrics["growth_metrics"].append({
                "turn_idx": turn_idx,
                "growth": growth
            })
            dialogue_metrics["cumulative_metrics"].append({
                "turn_idx": turn_idx,
                "counts": cumulative_counts,
                "normalized_counts": normalized_counts
            })
            # Update previous turn count for next iteration
            prev_turn_count = cumulative_counts.copy()
        
        # Calculate final metrics for the dialogue
        if dialogue_metrics["cumulative_metrics"]:
            max_possible_propositions = 37 ##calculate_theoretical_maximum() for WTD  
 
            for category in normalized_counts:
                raw = cumulative_counts[category]
                normalized = normalized_counts[category]
                expected = raw / max_possible_propositions

                if not abs(normalized - expected) < 1e-6:
                    print(f"[WARNING] Normalized count mismatch in dialogue {dialogue_id}, category: {category}")
                    print(f"  Raw: {raw}, Normalized: {normalized:.6f}, Expected: {expected:.6f}")
 
            final_counts = dialogue_metrics["cumulative_metrics"][-1]["counts"]
            
            dialogue_metrics["final_total"] = final_counts["total"]
            dialogue_metrics["final_by_category"] = {
                "equality": final_counts["equality"],
                "inequality": final_counts["inequality"],
                "order": final_counts["order"]
            }
            
            # Calculate average growth rate
            if len(dialogue_metrics["growth_metrics"]) > 1:
                total_growth = dialogue_metrics["growth_metrics"][1:]   
                avg_growth = sum(g["growth"]["total"] for g in total_growth) / len(total_growth)
                dialogue_metrics["avg_growth_rate"] = avg_growth
            else:
                dialogue_metrics["avg_growth_rate"] = 0
        
        model_metrics[dialogue_id] = dialogue_metrics
    
    return model_metrics

def compute_accuracy_adjusted_metrics_with_sem(friction_metrics, parsed_logs, correct_weights=None):
    """
    Compute accuracy-adjusted metrics by combining the common ground metrics
    with error analysis from identify_incorrect_relations. Includes standard error of the mean.
    
    Args:
        friction_metrics: Dictionary of metrics computed for each model
        parsed_logs: Dictionary of parsed logs for error analysis
        correct_weights: Dictionary of correct block weights
        
    Returns:
        Dictionary with enhanced metrics for each model, including SEM
    """
    # First, get the error analysis
    error_analysis = identify_incorrect_relations(parsed_logs, correct_weights)
    
    # Initialize results dictionary
    enhanced_metrics = {}
    aggregated_metrics = {}  # Separate dictionary for aggregated metrics
    
    # Process each model
    for model_name, model_metrics in friction_metrics.items():
        if model_name not in error_analysis:
            continue
            
        model_errors = error_analysis[model_name]
        enhanced_model_metrics = {}
        
        # Initialize dictionaries for collecting values for SEM calculation
        all_values = {
            "final_total": [],
            "accuracy_adjusted_total": [],
            "error_free_relations": [],
            "accuracy_to_quantity_ratio": [],
            "error_weighted_growth": [],
            "incorrect_percentage": []
        }
        
        # Process each dialogue
        for dialogue_id, dialogue_metrics in model_metrics.items():
            # Get final total relations
            final_total = dialogue_metrics.get("final_total", 0)
            
            # Find corresponding error information
            error_info = None
            for trial in model_errors.get("individual_trials", []):
                # In practice, you might need more specific matching logic
                error_info = trial
                break
                
            if error_info:
                # Calculate enhanced metrics
                incorrect_percentage = error_info.get("incorrect_percentage", 0)
                incorrect_relations = error_info.get("incorrect_relations", 0)
                
                # 1. Accuracy-adjusted total
                accuracy_adjusted_total = final_total * (1 - (incorrect_percentage / 100))
                
                # 2. Error-free relations count
                error_free_relations = final_total - incorrect_relations
                
                # 3. Accuracy-to-quantity ratio (higher is better)
                accuracy_to_quantity_ratio = (error_free_relations ** 2) / final_total if final_total > 0 else 0
                
                # 4. Error-weighted growth rate
                avg_growth_rate = dialogue_metrics.get("avg_growth_rate", 0)
                error_weighted_growth = avg_growth_rate * (1 - (incorrect_percentage / 100))
                
                # Store enhanced metrics
                enhanced_model_metrics[dialogue_id] = {
                    "final_total": final_total,
                    "accuracy_adjusted_total": accuracy_adjusted_total,
                    "error_free_relations": error_free_relations,
                    "accuracy_to_quantity_ratio": accuracy_to_quantity_ratio,
                    "error_weighted_growth": error_weighted_growth,
                    "incorrect_percentage": incorrect_percentage
                }
                
                # Add error type distribution if available
                if "errors_by_type" in error_info:
                    total_errors = sum(error_info["errors_by_type"].values())
                    if total_errors > 0:
                        error_distribution = {
                            category: errors / total_errors 
                            for category, errors in error_info["errors_by_type"].items()
                        }
                        enhanced_model_metrics[dialogue_id]["error_distribution"] = error_distribution
                
                # Collect values for SEM calculation - THIS IS MOVED INSIDE THE DIALOGUE LOOP
                all_values["final_total"].append(final_total)
                all_values["accuracy_adjusted_total"].append(accuracy_adjusted_total)
                all_values["error_free_relations"].append(error_free_relations)
                all_values["accuracy_to_quantity_ratio"].append(accuracy_to_quantity_ratio)
                all_values["error_weighted_growth"].append(error_weighted_growth)
                all_values["incorrect_percentage"].append(incorrect_percentage)
        
        # Store model metrics
        enhanced_metrics[model_name] = enhanced_model_metrics
        
        # Calculate aggregated metrics with SEM
        if all_values["final_total"]:  # Check if we have any values
            aggregated = {}
            
            # Calculate mean and SEM for each metric
            for metric_key, values in all_values.items():
                if values:
                    mean_value = np.mean(values)
            
                    sem_value = np.std(values) /  np.sqrt(len(values)) if len(values) > 1 else 0
                    
                    aggregated[f"avg_{metric_key}"] = mean_value
 
                    aggregated[f"sem_{metric_key}"] = sem_value
            
            # Store aggregated metrics
            aggregated_metrics[model_name] = aggregated
    
    # Add aggregated metrics to the result
    for model_name, aggregated in aggregated_metrics.items():
        enhanced_metrics[f"{model_name}_aggregated"] = aggregated
    
    return enhanced_metrics


 
def compute_deli_metrics(all_models_data):
    """
    Compute evaluation metrics for all models in the DELI dataset.
    
    Args:
        all_models_data: Dictionary with model names as keys and dialogue data as values
        
    Returns:
        Dictionary with computed metrics (mean and std) for each model
    """
    metrics = {}
    missing_data_counts = defaultdict(int)
    
    for model_name, model_data in all_models_data.items():
        model_metrics = {
            'final_solution_accuracy': [],
            'fine_grained_scores': [],
            'performance_gains': [],
            'initial_solution_diversity': [],
            'discussion_solution_diversity': [],
            'unique_transitions': [],
            'stuck_transitions': [],
            'circular_transitions': [],
            'participant_correct_solutions': defaultdict(list),  # Track per participant correct solutions
            'total_correct_solutions': 0,  # Add this line to track total correct solutions
            'total_solutions': 0  # Add this line to track total solutions
        }
        
        for dialogue_id, dialogue_data in model_data.items():
            # Check for missing data
            if not dialogue_data:
                missing_data_counts[f"{model_name}_missing_dialogue"] += 1
                logger.warning(f"Missing dialogue data for {dialogue_id} in model {model_name}")
                continue
                
            if 'turns' not in dialogue_data or not dialogue_data['turns']:
                missing_data_counts[f"{model_name}_missing_turns"] += 1
                logger.warning(f"Missing turns data for dialogue {dialogue_id} in model {model_name}")
                continue
                
            try:
                # Get the last turn data
                last_turn = dialogue_data['turns'][-1]
                if 'parsed_gpt_response' not in last_turn:
                    missing_data_counts[f"{model_name}_missing_parsed_gpt_response"] += 1
                    logger.warning(f"Missing parsed_gpt_response in last turn for dialogue {dialogue_id} in model {model_name}")
                    continue
                
                last_turn_data = last_turn['parsed_gpt_response']
                
                # 1. Final Solution Accuracy
                final_solution = last_turn_data.get('final_submission_mapped')
                if final_solution is None:
                    missing_data_counts[f"{model_name}_missing_final_submission"] += 1
                    logger.warning(f"Missing final_submission_mapped for dialogue {dialogue_id} in model {model_name}")
                    continue
                    
                is_correct = final_solution == 'OV'
                model_metrics['final_solution_accuracy'].append(int(is_correct))
                
                # 2. Fine-grained Scoring
                score = compute_fine_grained_score(final_solution)
                model_metrics['fine_grained_scores'].append(score)
                
                # 3. Performance Gain
                initial_solutions = dialogue_data.get('initial_solutions')
                if not initial_solutions:
                    missing_data_counts[f"{model_name}_missing_initial_solutions"] += 1
                    logger.info(f"Missing initial_solutions for dialogue {dialogue_id} in model {model_name}")
                else:
                    gain = compute_performance_gain(initial_solutions, final_solution)
                    if gain is not None:
                        model_metrics['performance_gains'].append(gain)
                
                # 4. Initial Solution Diversity
                if initial_solutions:
                    diversity = len(set(map_initial_solutions_to_framework(initial_solutions).values()))
                    model_metrics['initial_solution_diversity'].append(diversity)
                
                # 5. Discussion Solution Diversity
                all_solutions = set()
                
                # 6-8. Solution Transitions
                transitions = extract_solution_transitions(dialogue_data['turns'])
                
                # Track participant correct solutions across turns
                for turn_idx, turn in enumerate(dialogue_data['turns']):
                    if 'parsed_gpt_response' not in turn:
                        missing_data_counts[f"{model_name}_missing_turn_parsed_response"] += 1
                        logger.info(f"Missing parsed_gpt_response in turn {turn_idx} for dialogue {dialogue_id} in model {model_name}")
                        continue
                        
                    # Track solution diversity
                    if 'solution_mappings' in turn['parsed_gpt_response']:
                        solutions = turn['parsed_gpt_response']['solution_mappings'].values()
                        all_solutions.update(solutions)
                    
                    # Track participant correct solutions
                    if 'solution_mappings' in turn['parsed_gpt_response']:
                        for participant, solution in turn['parsed_gpt_response']['solution_mappings'].items():
                            is_correct = solution == 'OV'
                            model_metrics['participant_correct_solutions'][participant].append(int(is_correct))
                            
                            # Track total solutions and correct solutions
                            model_metrics['total_solutions'] += 1
                            if is_correct:
                                model_metrics['total_correct_solutions'] += 1
                
                model_metrics['discussion_solution_diversity'].append(len(all_solutions))
                
                # Analyze transitions
                unique_transitions, stuck_transitions, circular_transitions = analyze_transitions(transitions)
                model_metrics['unique_transitions'].append(unique_transitions)
                model_metrics['stuck_transitions'].append(stuck_transitions)
                model_metrics['circular_transitions'].append(circular_transitions)
                
            except Exception as e:
                logger.error(f"Error processing dialogue {dialogue_id} for model {model_name}: {e}", exc_info=True)
                continue
        
        # Calculate means and standard deviations for the model
        # Calculate means and standard error of the mean (SEM) for the model
        metrics[model_name] = {
            'final_solution_accuracy': {
                'mean': safe_mean(model_metrics['final_solution_accuracy']),
                'sem': safe_std(model_metrics['final_solution_accuracy']) / math.sqrt(len(model_metrics['final_solution_accuracy'])) if model_metrics['final_solution_accuracy'] else 0
            },
            'fine_grained_score': {
                'mean': safe_mean(model_metrics['fine_grained_scores']),
                'sem': safe_std(model_metrics['fine_grained_scores']) / math.sqrt(len(model_metrics['fine_grained_scores'])) if model_metrics['fine_grained_scores'] else 0
            },
            'performance_gain': {
                'mean': safe_mean(model_metrics['performance_gains']),
                'sem': safe_std(model_metrics['performance_gains']) / math.sqrt(len(model_metrics['performance_gains'])) if model_metrics['performance_gains'] else 0
            },
            'initial_solution_diversity': {
                'mean': safe_mean(model_metrics['initial_solution_diversity']),
                'sem': safe_std(model_metrics['initial_solution_diversity']) / math.sqrt(len(model_metrics['initial_solution_diversity'])) if model_metrics['initial_solution_diversity'] else 0
            },
            'discussion_solution_diversity': {
                'mean': safe_mean(model_metrics['discussion_solution_diversity']),
                'sem': safe_std(model_metrics['discussion_solution_diversity']) / math.sqrt(len(model_metrics['discussion_solution_diversity'])) if model_metrics['discussion_solution_diversity'] else 0
            },
            'unique_transitions': {
                'mean': safe_mean(model_metrics['unique_transitions']),
                'sem': safe_std(model_metrics['unique_transitions']) / math.sqrt(len(model_metrics['unique_transitions'])) if model_metrics['unique_transitions'] else 0
            },
            'stuck_transitions': {
                'mean': safe_mean(model_metrics['stuck_transitions']),
                'sem': safe_std(model_metrics['stuck_transitions']) / math.sqrt(len(model_metrics['stuck_transitions'])) if model_metrics['stuck_transitions'] else 0
            },
            'circular_transitions': {
                'mean': safe_mean(model_metrics['circular_transitions']),
                'sem': safe_std(model_metrics['circular_transitions']) / math.sqrt(len(model_metrics['circular_transitions'])) if model_metrics['circular_transitions'] else 0
            },
            'participant_correct_solutions': {},
            # Add total correct solutions statistics
            'total_correct_solutions_count': model_metrics['total_correct_solutions'],
            'total_solutions_count': model_metrics['total_solutions'],
            'total_correct_solutions_percentage': (model_metrics['total_correct_solutions'] / model_metrics['total_solutions'] * 100) if model_metrics['total_solutions'] > 0 else 0
        }
    
    # Log missing data summary
    for key, count in missing_data_counts.items():
        logger.info(f"{key}: {count} instances")
    
    return metrics

def compute_fine_grained_score(solution_mapping):
    """
    Compute fine-grained score based on the 0.25-point system.
    
    Args:
        solution_mapping: String representing the solution mapping (e.g., 'OV', 'EV', etc.)
        
    Returns:
        Float score between 0 and 1
    """
    if solution_mapping is None:
        return 0.0
    
    score = 0.0
    
    # Check for inclusion of target cards
    if 'O' in solution_mapping:
        score += 0.25
    if 'V' in solution_mapping:
        score += 0.25
    
    # Check for exclusion of unnecessary cards
    if 'E' not in solution_mapping:
        score += 0.25
    if 'C' not in solution_mapping:
        score += 0.25
    
    return score

def map_initial_solutions_to_framework(initial_solutions):
    """
    Map initial solutions to the CEOV framework.
    
    Args:
        initial_solutions: Dictionary mapping participant names to lists of selected cards
        
    Returns:
        Dictionary mapping participant names to solution framework strings
    """
    mapped_solutions = {}
    
    # Handle different formats of initial_solutions
    if isinstance(initial_solutions, str):
        try:
            import json
            initial_solutions = json.loads(initial_solutions.replace("'", '"'))
        except:
            logger.warning(f"Could not parse initial_solutions string: {initial_solutions}")
            return mapped_solutions
    
    if not isinstance(initial_solutions, dict):
        logger.warning(f"Initial solutions is not a dictionary: {type(initial_solutions)}")
        return mapped_solutions
    
    for participant, cards in initial_solutions.items():
        if not cards:  # Skip if no cards
            continue
            
        solution = ''
        
        # Check for consonant
        if any(card.isalpha() and card.upper() not in 'AEIOU' for card in cards):
            solution += 'C'
        
        # Check for even number
        if any(card.isdigit() and int(card) % 2 == 0 for card in cards):
            solution += 'E'
        
        # Check for odd number
        if any(card.isdigit() and int(card) % 2 == 1 for card in cards):
            solution += 'O'
        
        # Check for vowel
        if any(card.isalpha() and card.upper() in 'AEIOU' for card in cards):
            solution += 'V'
        
        mapped_solutions[participant] = solution if solution else 'none'
    
    return mapped_solutions

def compute_performance_gain(initial_solutions, final_solution):
    """
    Compute performance gain from initial to final solutions.
    
    Args:
        initial_solutions: Dictionary mapping participant names to lists of selected cards
        final_solution: String representing the final solution mapping
        
    Returns:
        Float representing the average performance gain
    """
    if not initial_solutions or final_solution is None:
        return None
    
    final_score = compute_fine_grained_score(final_solution)
    
    try:
        # Map initial solutions to framework and compute scores
        mapped_solutions = map_initial_solutions_to_framework(initial_solutions)
        
        # Calculate initial scores
        initial_scores = [compute_fine_grained_score(solution) for solution in mapped_solutions.values()]
        
        if not initial_scores:
            return None
        
        # Calculate average initial score
        avg_initial_score = sum(initial_scores) / len(initial_scores)
        
        # Calculate gain
        return final_score - avg_initial_score
    except Exception as e:
        logger.error(f"Error computing performance gain: {e}", exc_info=True)
        return None

def extract_solution_transitions(turns):
    """
    Extract solution transitions for each participant across turns.
    
    Args:
        turns: List of turn data
        
    Returns:
        Dictionary mapping participants to lists of solution sequences
    """
    participant_solutions = defaultdict(list)
    
    for turn in turns:
        if 'parsed_gpt_response' not in turn:
            continue
            
        solution_mappings = turn['parsed_gpt_response'].get('solution_mappings', {})
        
        for participant, solution in solution_mappings.items():
            if solution:  # Skip empty solutions
                participant_solutions[participant].append(solution)
    
    return participant_solutions

def analyze_transitions(participant_solutions):
    """
    Analyze solution transitions to count unique, stuck, and circular transitions.
    
    Args:
        participant_solutions: Dictionary mapping participants to lists of solution sequences
        
    Returns:
        Tuple of (unique transitions count, stuck transitions count, circular transitions count)
    """
    unique_transitions = set()
    stuck_transitions = 0
    circular_transitions = 0
    
    for participant, solutions in participant_solutions.items():
        if len(solutions) < 3:  # Need at least 3 solutions for a transition triple
            continue
            
        # Create transition triples
        for i in range(len(solutions) - 2):
            triple = f"{solutions[i]}-{solutions[i+1]}-{solutions[i+2]}"
            unique_transitions.add(triple)
            
            # Check for stuck transitions
            if solutions[i] == solutions[i+1] == solutions[i+2]:
                stuck_transitions += 1
            
            # Check for circular transitions
            if solutions[i] == solutions[i+2] and solutions[i] != solutions[i+1]:
                circular_transitions += 1
    
    return len(unique_transitions), stuck_transitions, circular_transitions

def safe_mean(values):
    """Calculate mean safely even with empty lists."""
    return float(np.mean(values)) if values else 0.0

def safe_std(values):
    """Calculate standard deviation safely even with empty lists."""
    return float(np.std(values)) if len(values) > 1 else 0.0

def print_metrics(metrics):
    """Print metrics in a readable format."""
    print("Model Performance Metrics:\n")
    
    for model_name, model_metrics in metrics.items():
        print(f"Model: {model_name}")
        print(f"  Final Solution Accuracy: {model_metrics['final_solution_accuracy']['mean']:.2f} ± {model_metrics['final_solution_accuracy']['sem']:.2f}")
        print(f"  Fine-grained Score: {model_metrics['fine_grained_score']['mean']:.2f} ± {model_metrics['fine_grained_score']['sem']:.2f}")
        print(f"  Performance Gain: {model_metrics['performance_gain']['mean']:.2f} ± {model_metrics['performance_gain']['sem']:.2f}")
        print(f"  Initial Solution Diversity: {model_metrics['initial_solution_diversity']['mean']:.2f} ± {model_metrics['initial_solution_diversity']['sem']:.2f}")
        print(f"  Discussion Solution Diversity: {model_metrics['discussion_solution_diversity']['mean']:.2f} ± {model_metrics['discussion_solution_diversity']['sem']:.2f}")
        print(f"  Unique Transitions: {model_metrics['unique_transitions']['mean']:.2f} ± {model_metrics['unique_transitions']['sem']:.2f}")
        print(f"  Stuck Transitions: {model_metrics['stuck_transitions']['mean']:.2f} ± {model_metrics['stuck_transitions']['sem']:.2f}")
        print(f"  Circular Transitions: {model_metrics['circular_transitions']['mean']:.2f} ± {model_metrics['circular_transitions']['sem']:.2f}")
        
        print(f"  Total Solutions: {model_metrics['total_solutions_count']} (Correct: {model_metrics['total_correct_solutions_count']}, {model_metrics['total_correct_solutions_percentage']:.2f}%)")
 
    return None
 
