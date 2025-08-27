import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import json
from difflib import SequenceMatcher
import re

class EvaluationFramework:
    def __init__(self, ground_truth_df: pd.DataFrame):
        """
        Initialize with manually verified ground truth data.
        
        Args:
            ground_truth_df: DataFrame with columns ['age', 'recommended_treatment', 
                           'primary_diagnosis', 'icd_10_code', 'icd_10_description']
        """
        self.ground_truth = ground_truth_df
        self.results = {}
        
        # ICD-10 code structure for category-level matching
        self.icd_10_categories = {
            'A00-B99': 'Infectious and parasitic diseases',
            'C00-D49': 'Neoplasms',
            'D50-D89': 'Diseases of blood and immune system',
            'E00-E89': 'Endocrine, nutritional and metabolic diseases',
            'F01-F99': 'Mental and behavioural disorders',
            'G00-G99': 'Diseases of the nervous system',
            'H00-H59': 'Diseases of the eye and adnexa',
            'H60-H95': 'Diseases of the ear and mastoid process',
            'I00-I99': 'Diseases of the circulatory system',
            'J00-J99': 'Diseases of the respiratory system',
            'K00-K95': 'Diseases of the digestive system',
            'L00-L99': 'Diseases of the skin and subcutaneous tissue',
            'M00-M99': 'Diseases of the musculoskeletal system',
            'N00-N99': 'Diseases of the genitourinary system',
            'O00-O9A': 'Pregnancy, childbirth and the puerperium',
            'P00-P96': 'Perinatal conditions',
            'Q00-Q99': 'Congenital malformations and chromosomal abnormalities',
            'R00-R99': 'Symptoms and abnormal findings',
            'S00-T88': 'Injury, poisoning and external causes',
            'V00-Y99': 'External causes of morbidity',
            'Z00-Z99': 'Health status and health services contact'
        }
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using sequence matching."""
        if not text1 or not text2:
            return 0.0
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def get_icd_category(self, icd_code: str) -> str:
        """Get the ICD-10 category for a given code."""
        if not icd_code or icd_code == "N/A":
            return "Unknown"
        
        # Extract the letter and first two digits
        match = re.match(r'^([A-Z])(\d{2})', icd_code)
        if not match:
            return "Unknown"
        
        letter = match.group(1)
        number = int(match.group(2))
        
        for category_range, description in self.icd_10_categories.items():
            start_code, end_code = category_range.split('-')
            start_letter = start_code[0]
            start_num = int(start_code[1:3]) if len(start_code) > 1 else 0
            end_letter = end_code[0]
            end_num = int(end_code[1:3]) if len(end_code) > 1 else 99
            
            if (letter == start_letter and number >= start_num) and \
               (letter < end_letter or (letter == end_letter and number <= end_num)):
                return description
        
        return "Unknown"
    
    def evaluate_field_accuracy(self, predicted: List[str], actual: List[str], 
                               field_name: str, similarity_threshold: float = 0.8) -> Dict[str, float]:
        """
        Evaluate accuracy for a specific field.
        
        Returns:
            Dict with exact_match, partial_match, precision, recall, f1_score
        """
        if len(predicted) != len(actual):
            raise ValueError(f"Predicted and actual lists must have same length for field {field_name}")
        
        exact_matches = 0
        partial_matches = 0
        similarities = []
        
        for pred, act in zip(predicted, actual):
            pred_str = str(pred).strip() if pred is not None else ""
            act_str = str(act).strip() if act is not None else ""
            
            # Exact match
            if pred_str.lower() == act_str.lower():
                exact_matches += 1
                partial_matches += 1
                similarities.append(1.0)
            else:
                # Partial match using similarity
                similarity = self.calculate_text_similarity(pred_str, act_str)
                similarities.append(similarity)
                if similarity >= similarity_threshold:
                    partial_matches += 1
        
        total = len(predicted)
        exact_accuracy = exact_matches / total if total > 0 else 0
        partial_accuracy = partial_matches / total if total > 0 else 0
        avg_similarity = np.mean(similarities) if similarities else 0
        
        return {
            'exact_match_accuracy': exact_accuracy,
            'partial_match_accuracy': partial_accuracy,
            'average_similarity': avg_similarity,
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'total_samples': total
        }
    
    def evaluate_icd_codes(self, predicted_codes: List[str], actual_codes: List[str]) -> Dict[str, float]:
        """Evaluate ICD-10 code accuracy at multiple levels."""
        exact_matches = 0
        category_matches = 0
        chapter_matches = 0
        
        for pred, actual in zip(predicted_codes, actual_codes):
            pred_str = str(pred).strip() if pred is not None else ""
            actual_str = str(actual).strip() if actual is not None else ""
            
            # Exact match
            if pred_str == actual_str:
                exact_matches += 1
                category_matches += 1
                chapter_matches += 1
            else:
                # Category level match (first 3 characters)
                if len(pred_str) >= 3 and len(actual_str) >= 3:
                    if pred_str[:3] == actual_str[:3]:
                        category_matches += 1
                        chapter_matches += 1
                    else:
                        # Chapter level match (ICD-10 category)
                        pred_category = self.get_icd_category(pred_str)
                        actual_category = self.get_icd_category(actual_str)
                        if pred_category == actual_category and pred_category != "Unknown":
                            chapter_matches += 1
        
        total = len(predicted_codes)
        return {
            'exact_accuracy': exact_matches / total if total > 0 else 0,
            'category_accuracy': category_matches / total if total > 0 else 0,
            'chapter_accuracy': chapter_matches / total if total > 0 else 0,
            'exact_matches': exact_matches,
            'category_matches': category_matches,
            'chapter_matches': chapter_matches,
            'total_samples': total
        }
    
    def evaluate_extractor(self, extractor_results: pd.DataFrame, extractor_name: str, 
                          processing_time: float = None, api_cost: float = None) -> Dict[str, Any]:
        """
        Evaluate a single extractor against ground truth.
        
        Args:
            extractor_results: DataFrame with same structure as ground_truth
            extractor_name: Name of the extractor being evaluated
            processing_time: Total processing time in seconds
            api_cost: Total API cost in USD (if applicable)
        """
        results = {
            'extractor_name': extractor_name,
            'processing_time': processing_time,
            'api_cost': api_cost,
            'field_evaluations': {}
        }
        
        # Evaluate each field
        fields_to_evaluate = ['age', 'recommended_treatment', 'primary_diagnosis', 'icd_10_description']
        
        for field in fields_to_evaluate:
            if field in extractor_results.columns and field in self.ground_truth.columns:
                predicted = extractor_results[field].tolist()
                actual = self.ground_truth[field].tolist()
                
                field_results = self.evaluate_field_accuracy(predicted, actual, field)
                results['field_evaluations'][field] = field_results
        
        # Special evaluation for ICD-10 codes
        if 'icd_10_code' in extractor_results.columns and 'icd_10_code' in self.ground_truth.columns:
            predicted_codes = extractor_results['icd_10_code'].tolist()
            actual_codes = self.ground_truth['icd_10_code'].tolist()
            
            icd_results = self.evaluate_icd_codes(predicted_codes, actual_codes)
            results['field_evaluations']['icd_10_code'] = icd_results
        
        # Calculate overall metrics
        exact_scores = [results['field_evaluations'][field]['exact_match_accuracy'] 
                       for field in results['field_evaluations']]
        partial_scores = [results['field_evaluations'][field].get('partial_match_accuracy', 
                         results['field_evaluations'][field].get('exact_accuracy', 0))
                         for field in results['field_evaluations']]
        
        results['overall_exact_accuracy'] = np.mean(exact_scores) if exact_scores else 0
        results['overall_partial_accuracy'] = np.mean(partial_scores) if partial_scores else 0
        
        return results
    
    def compare_extractors(self, baseline_results: pd.DataFrame, openai_results: pd.DataFrame,
                          baseline_time: float = None, openai_time: float = None,
                          openai_cost: float = None) -> Dict[str, Any]:
        """
        Compare baseline and OpenAI extractors.
        
        Returns comprehensive comparison metrics.
        """
        baseline_eval = self.evaluate_extractor(baseline_results, "Baseline (Regex)", 
                                               baseline_time, 0)
        openai_eval = self.evaluate_extractor(openai_results, "OpenAI API", 
                                            openai_time, openai_cost)
        
        comparison = {
            'baseline': baseline_eval,
            'openai': openai_eval,
            'comparison_metrics': {}
        }
        
        # Performance comparison
        comparison['comparison_metrics']['accuracy_improvement'] = \
            openai_eval['overall_exact_accuracy'] - baseline_eval['overall_exact_accuracy']
        
        # Time comparison
        if baseline_time and openai_time:
            comparison['comparison_metrics']['time_ratio'] = openai_time / baseline_time
            comparison['comparison_metrics']['time_difference'] = openai_time - baseline_time
        
        # Cost per accuracy improvement
        if openai_cost and comparison['comparison_metrics']['accuracy_improvement'] > 0:
            comparison['comparison_metrics']['cost_per_accuracy_point'] = \
                openai_cost / comparison['comparison_metrics']['accuracy_improvement']
        
        # Field-by-field comparison
        field_comparisons = {}
        for field in baseline_eval['field_evaluations']:
            if field in openai_eval['field_evaluations']:
                baseline_acc = baseline_eval['field_evaluations'][field]['exact_match_accuracy']
                openai_acc = openai_eval['field_evaluations'][field]['exact_match_accuracy']
                field_comparisons[field] = {
                    'baseline_accuracy': baseline_acc,
                    'openai_accuracy': openai_acc,
                    'improvement': openai_acc - baseline_acc
                }
        
        comparison['comparison_metrics']['field_comparisons'] = field_comparisons
        
        return comparison
    
    def generate_detailed_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate a detailed text report of the comparison."""
        report = []
        report.append("=" * 80)
        report.append("MEDICAL TRANSCRIPTION EXTRACTOR EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall Performance Summary
        report.append("OVERALL PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        
        baseline = comparison_results['baseline']
        openai = comparison_results['openai']
        
        report.append(f"Baseline Extractor (Regex):")
        report.append(f"  Overall Exact Accuracy: {baseline['overall_exact_accuracy']:.1%}")
        report.append(f"  Overall Partial Accuracy: {baseline['overall_partial_accuracy']:.1%}")
        if baseline['processing_time']:
            report.append(f"  Processing Time: {baseline['processing_time']:.2f}s")
        report.append("")
        
        report.append(f"OpenAI API Extractor:")
        report.append(f"  Overall Exact Accuracy: {openai['overall_exact_accuracy']:.1%}")
        report.append(f"  Overall Partial Accuracy: {openai['overall_partial_accuracy']:.1%}")
        if openai['processing_time']:
            report.append(f"  Processing Time: {openai['processing_time']:.2f}s")
        if openai['api_cost']:
            report.append(f"  API Cost: ${openai['api_cost']:.4f}")
        report.append("")
        
        # Improvement Analysis
        metrics = comparison_results['comparison_metrics']
        report.append("IMPROVEMENT ANALYSIS:")
        report.append("-" * 40)
        report.append(f"Accuracy Improvement: {metrics['accuracy_improvement']:+.1%}")
        
        if 'time_ratio' in metrics:
            report.append(f"Time Ratio (OpenAI/Baseline): {metrics['time_ratio']:.1f}x")
        
        if 'cost_per_accuracy_point' in metrics:
            report.append(f"Cost per Accuracy Point: ${metrics['cost_per_accuracy_point']:.4f}")
        report.append("")
        
        # Field-by-Field Analysis
        report.append("FIELD-BY-FIELD ANALYSIS:")
        report.append("-" * 40)
        
        for field, comparison in metrics['field_comparisons'].items():
            report.append(f"{field.replace('_', ' ').title()}:")
            report.append(f"  Baseline: {comparison['baseline_accuracy']:.1%}")
            report.append(f"  OpenAI:   {comparison['openai_accuracy']:.1%}")
            report.append(f"  Improvement: {comparison['improvement']:+.1%}")
            report.append("")
        
        # Detailed Field Metrics
        report.append("DETAILED FIELD METRICS:")
        report.append("-" * 40)
        
        for extractor_name, extractor_data in [("Baseline", baseline), ("OpenAI", openai)]:
            report.append(f"\n{extractor_name} Extractor Details:")
            for field, metrics in extractor_data['field_evaluations'].items():
                report.append(f"  {field.replace('_', ' ').title()}:")
                report.append(f"    Exact Matches: {metrics.get('exact_matches', 0)}/{metrics.get('total_samples', 0)}")
                if 'partial_matches' in metrics:
                    report.append(f"    Partial Matches: {metrics['partial_matches']}/{metrics['total_samples']}")
                if 'average_similarity' in metrics:
                    report.append(f"    Avg Similarity: {metrics['average_similarity']:.1%}")
        
        # Recommendations
        report.append("\nRECOMMENDations:")
        report.append("-" * 40)
        
        if metrics['accuracy_improvement'] > 0.1:
            report.append("✓ OpenAI API provides significant accuracy improvement")
        elif metrics['accuracy_improvement'] > 0.05:
            report.append("⚠ OpenAI API provides moderate accuracy improvement")
        else:
            report.append("✗ OpenAI API provides minimal accuracy improvement")
        
        if 'cost_per_accuracy_point' in metrics:
            if metrics['cost_per_accuracy_point'] < 0.01:
                report.append("✓ Cost-effective accuracy improvement")
            else:
                report.append("⚠ Consider cost vs. accuracy trade-off")
        
        if 'time_ratio' in metrics:
            if metrics['time_ratio'] < 2:
                report.append("✓ Reasonable processing time increase")
            else:
                report.append("⚠ Significant processing time increase")
        
        return "\n".join(report)
    
    def save_detailed_comparison(self, comparison_results: Dict[str, Any], filename: str = "evaluation_report.txt"):
        """Save detailed comparison report to file."""
        report = self.generate_detailed_report(comparison_results)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Detailed evaluation report saved to {filename}")
    
    def export_metrics_csv(self, comparison_results: Dict[str, Any], filename: str = "evaluation_metrics.csv"):
        """Export metrics to CSV for further analysis."""
        rows = []
        
        for extractor_name in ['baseline', 'openai']:
            extractor_data = comparison_results[extractor_name]
            
            for field, metrics in extractor_data['field_evaluations'].items():
                row = {
                    'extractor': extractor_data['extractor_name'],
                    'field': field,
                    'exact_accuracy': metrics.get('exact_match_accuracy', metrics.get('exact_accuracy', 0)),
                    'partial_accuracy': metrics.get('partial_match_accuracy', 0),
                    'average_similarity': metrics.get('average_similarity', 0),
                    'exact_matches': metrics.get('exact_matches', 0),
                    'total_samples': metrics.get('total_samples', 0)
                }
                rows.append(row)
        
        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(filename, index=False)
        print(f"Evaluation metrics exported to {filename}")