import pandas as pd
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re

class EvaluationMetrics:
    """
    Evaluation metrics for comparing medical information extraction approaches.
    """
    
    def __init__(self):
        self.metrics_cache = {}
    
    def calculate_exact_match_accuracy(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate exact match accuracy between predictions and ground truth."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        matches = sum(1 for p, gt in zip(predictions, ground_truth) if p.strip().lower() == gt.strip().lower())
        return matches / len(predictions)
    
    def calculate_partial_match_score(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate partial match score using token overlap."""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        total_score = 0
        for pred, gt in zip(predictions, ground_truth):
            pred_tokens = set(re.findall(r'\b\w+\b', pred.lower()))
            gt_tokens = set(re.findall(r'\b\w+\b', gt.lower()))
            
            if len(gt_tokens) == 0:
                score = 1.0 if len(pred_tokens) == 0 else 0.0
            else:
                intersection = len(pred_tokens & gt_tokens)
                union = len(pred_tokens | gt_tokens)
                score = intersection / union if union > 0 else 0.0
            
            total_score += score
        
        return total_score / len(predictions)
    
    def calculate_icd10_accuracy(self, predicted_codes: List[str], ground_truth_codes: List[str]) -> Dict[str, float]:
        """Calculate ICD-10 code accuracy at different levels of specificity."""
        if len(predicted_codes) != len(ground_truth_codes):
            raise ValueError("Predictions and ground truth must have the same length")
        
        exact_matches = 0
        category_matches = 0  # First 3 characters
        chapter_matches = 0   # First character
        
        for pred, gt in zip(predicted_codes, ground_truth_codes):
            pred = pred.strip().upper()
            gt = gt.strip().upper()
            
            if pred == gt:
                exact_matches += 1
                category_matches += 1
                chapter_matches += 1
            elif len(pred) >= 3 and len(gt) >= 3 and pred[:3] == gt[:3]:
                category_matches += 1
                chapter_matches += 1
            elif len(pred) >= 1 and len(gt) >= 1 and pred[0] == gt[0]:
                chapter_matches += 1
        
        total = len(predicted_codes)
        return {
            "exact_match": exact_matches / total,
            "category_match": category_matches / total,
            "chapter_match": chapter_matches / total
        }
    
    def calculate_confidence_calibration(self, confidences: List[float], accuracies: List[bool]) -> Dict[str, float]:
        """Calculate confidence calibration metrics."""
        if len(confidences) != len(accuracies):
            raise ValueError("Confidences and accuracies must have the same length")
        
        # Convert to numpy arrays
        conf_array = np.array(confidences)
        acc_array = np.array(accuracies)
        
        # Calculate Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (conf_array > bin_lower) & (conf_array <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = acc_array[in_bin].mean()
                avg_confidence_in_bin = conf_array[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Calculate Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (conf_array > bin_lower) & (conf_array <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = acc_array[in_bin].mean()
                avg_confidence_in_bin = conf_array[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return {
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "average_confidence": conf_array.mean(),
            "average_accuracy": acc_array.mean()
        }
    
    def compare_extractors(self, baseline_results: pd.DataFrame, openai_results: pd.DataFrame, 
                          ground_truth: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare baseline and OpenAI extractors against ground truth.
        
        Args:
            baseline_results: Results from baseline extractor
            openai_results: Results from OpenAI extractor
            ground_truth: Ground truth annotations
            
        Returns:
            Comprehensive comparison metrics
        """
        
        comparison = {
            "baseline": {},
            "openai": {},
            "comparison": {}
        }
        
        # Fields to evaluate
        fields = ["age", "recommended_treatment", "primary_diagnosis", "icd_10_code"]
        
        for field in fields:
            if field in ground_truth.columns:
                # Baseline metrics
                baseline_pred = baseline_results[field].tolist()
                openai_pred = openai_results[field].tolist()
                gt = ground_truth[field].tolist()
                
                # Exact match accuracy
                baseline_exact = self.calculate_exact_match_accuracy(baseline_pred, gt)
                openai_exact = self.calculate_exact_match_accuracy(openai_pred, gt)
                
                # Partial match score
                baseline_partial = self.calculate_partial_match_score(baseline_pred, gt)
                openai_partial = self.calculate_partial_match_score(openai_pred, gt)
                
                comparison["baseline"][field] = {
                    "exact_match": baseline_exact,
                    "partial_match": baseline_partial
                }
                
                comparison["openai"][field] = {
                    "exact_match": openai_exact,
                    "partial_match": openai_partial
                }
                
                # ICD-10 specific metrics
                if field == "icd_10_code":
                    baseline_icd = self.calculate_icd10_accuracy(baseline_pred, gt)
                    openai_icd = self.calculate_icd10_accuracy(openai_pred, gt)
                    
                    comparison["baseline"][field].update(baseline_icd)
                    comparison["openai"][field].update(openai_icd)
                
                # Confidence calibration (if available)
                if "overall_confidence" in baseline_results.columns:
                    baseline_conf = baseline_results["overall_confidence"].tolist()
                    baseline_acc = [p.strip().lower() == g.strip().lower() for p, g in zip(baseline_pred, gt)]
                    baseline_cal = self.calculate_confidence_calibration(baseline_conf, baseline_acc)
                    comparison["baseline"][f"{field}_calibration"] = baseline_cal
                
                if "overall_confidence" in openai_results.columns:
                    openai_conf = openai_results["overall_confidence"].tolist()
                    openai_acc = [p.strip().lower() == g.strip().lower() for p, g in zip(openai_pred, gt)]
                    openai_cal = self.calculate_confidence_calibration(openai_conf, openai_acc)
                    comparison["openai"][f"{field}_calibration"] = openai_cal
        
        # Overall comparison
        comparison["comparison"]["winner_by_field"] = {}
        for field in fields:
            if field in comparison["baseline"] and field in comparison["openai"]:
                baseline_score = comparison["baseline"][field]["exact_match"]
                openai_score = comparison["openai"][field]["exact_match"]
                
                if openai_score > baseline_score:
                    comparison["comparison"]["winner_by_field"][field] = "openai"
                elif baseline_score > openai_score:
                    comparison["comparison"]["winner_by_field"][field] = "baseline"
                else:
                    comparison["comparison"]["winner_by_field"][field] = "tie"
        
        return comparison
    
    def generate_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        
        report = "Medical Information Extraction Evaluation Report\n"
        report += "=" * 60 + "\n\n"
        
        # Overall comparison
        report += "OVERALL COMPARISON\n"
        report += "-" * 20 + "\n"
        
        winners = comparison_results["comparison"]["winner_by_field"]
        baseline_wins = sum(1 for winner in winners.values() if winner == "baseline")
        openai_wins = sum(1 for winner in winners.values() if winner == "openai")
        ties = sum(1 for winner in winners.values() if winner == "tie")
        
        report += f"Baseline Extractor wins: {baseline_wins} fields\n"
        report += f"OpenAI Extractor wins: {openai_wins} fields\n"
        report += f"Ties: {ties} fields\n\n"
        
        # Detailed field comparison
        report += "DETAILED FIELD COMPARISON\n"
        report += "-" * 30 + "\n"
        
        for field, winner in winners.items():
            report += f"\n{field.upper()}:\n"
            
            if field in comparison_results["baseline"]:
                baseline_exact = comparison_results["baseline"][field]["exact_match"]
                baseline_partial = comparison_results["baseline"][field]["partial_match"]
                report += f"  Baseline: {baseline_exact:.3f} exact, {baseline_partial:.3f} partial\n"
            
            if field in comparison_results["openai"]:
                openai_exact = comparison_results["openai"][field]["exact_match"]
                openai_partial = comparison_results["openai"][field]["partial_match"]
                report += f"  OpenAI:   {openai_exact:.3f} exact, {openai_partial:.3f} partial\n"
            
            report += f"  Winner: {winner}\n"
        
        return report
