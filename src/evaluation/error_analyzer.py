import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import re
import json
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

class ErrorAnalyzer:
    def __init__(self, results_df: pd.DataFrame, ground_truth_df: pd.DataFrame):
        """
        Analyze discrepancies between extraction methods and ground truth.
        
        Args:
            results_df: DataFrame with extraction results (baseline or OpenAI)
            ground_truth_df: DataFrame with manually verified correct answers
        """
        self.results_df = results_df.copy()
        self.ground_truth_df = ground_truth_df.copy()
        self.error_analysis = {}
        self.field_errors = {}
        self.pattern_analysis = {}
        
        # Error type definitions
        self.error_types = {
            'missing': 'Information not extracted when it should be present',
            'incorrect': 'Wrong information extracted',
            'partial': 'Partially correct information extracted',
            'overextracted': 'Too much or unnecessary information extracted',
            'format_error': 'Correct information but wrong format',
            'none': 'No error - extraction is correct'
        }
        
        # Medical complexity indicators
        self.complexity_indicators = [
            'multiple', 'bilateral', 'chronic', 'acute', 'severe', 'moderate',
            'history of', 'previous', 'recurrent', 'complicated', 'underlying'
        ]
        
        # Ensure both DataFrames have compatible structure
        self._validate_dataframes()
    
    def _validate_dataframes(self):
        """Validate and align DataFrames for analysis."""
        # Ensure same length
        min_len = min(len(self.results_df), len(self.ground_truth_df))
        if len(self.results_df) != len(self.ground_truth_df):
            print(f"⚠️ Warning: DataFrame length mismatch. Using first {min_len} records.")
            self.results_df = self.results_df.iloc[:min_len].reset_index(drop=True)
            self.ground_truth_df = self.ground_truth_df.iloc[:min_len].reset_index(drop=True)
        
        # Identify common fields
        self.common_fields = [col for col in self.ground_truth_df.columns 
                             if col in self.results_df.columns and col != 'index']
        
        if not self.common_fields:
            raise ValueError("No common fields found between results and ground truth DataFrames")
        
        print(f"📊 Analyzing {len(self.results_df)} records across {len(self.common_fields)} fields")
        print(f"Fields: {', '.join(self.common_fields)}")
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two text strings."""
        if pd.isna(text1) or pd.isna(text2):
            return 0.0 if pd.isna(text1) != pd.isna(text2) else 1.0
        
        str1 = str(text1).strip().lower()
        str2 = str(text2).strip().lower()
        
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        return SequenceMatcher(None, str1, str2).ratio()
    
    def categorize_error(self, predicted: Any, actual: Any, field_name: str) -> Dict[str, Any]:
        """
        Categorize the type of error for a specific field.
        
        Returns:
            Dict with error type, severity, similarity, and details
        """
        pred_str = str(predicted).strip() if pd.notna(predicted) else ""
        actual_str = str(actual).strip() if pd.notna(actual) else ""
        
        # Handle 'Not specified' and similar placeholder values
        if pred_str.lower() in ['not specified', 'n/a', 'none', '']:
            pred_str = ""
        if actual_str.lower() in ['not specified', 'n/a', 'none', '']:
            actual_str = ""
        
        # Calculate similarity
        similarity = self.calculate_text_similarity(pred_str, actual_str)
        
        # Categorize error
        if not pred_str and not actual_str:
            return {
                'error_type': 'none',
                'severity': 0.0,
                'similarity': 1.0,
                'details': 'Both empty - no error'
            }
        elif not pred_str and actual_str:
            return {
                'error_type': 'missing',
                'severity': 1.0,
                'similarity': 0.0,
                'details': f"Failed to extract: '{actual_str}'"
            }
        elif pred_str and not actual_str:
            return {
                'error_type': 'overextracted',
                'severity': 0.6,
                'similarity': 0.0,
                'details': f"Extracted when shouldn't: '{pred_str}'"
            }
        elif similarity >= 0.95:
            return {
                'error_type': 'none',
                'severity': 0.0,
                'similarity': similarity,
                'details': 'Exact or near-exact match'
            }
        elif similarity >= 0.8:
            return {
                'error_type': 'partial',
                'severity': 0.2,
                'similarity': similarity,
                'details': f"Minor differences: '{pred_str}' vs '{actual_str}'"
            }
        elif similarity >= 0.5:
            return {
                'error_type': 'partial',
                'severity': 0.5,
                'similarity': similarity,
                'details': f"Partial match: '{pred_str}' vs '{actual_str}'"
            }
        elif similarity >= 0.2:
            return {
                'error_type': 'incorrect',
                'severity': 0.8,
                'similarity': similarity,
                'details': f"Mostly incorrect: '{pred_str}' vs '{actual_str}'"
            }
        else:
            return {
                'error_type': 'incorrect',
                'severity': 1.0,
                'similarity': similarity,
                'details': f"Completely wrong: '{pred_str}' vs '{actual_str}'"
            }
    
    def analyze_field_errors(self, field_name: str) -> Dict[str, Any]:
        """Comprehensive error analysis for a specific field."""
        if field_name not in self.common_fields:
            return {'error': f'Field {field_name} not available for analysis'}
        
        predicted = self.results_df[field_name].tolist()
        actual = self.ground_truth_df[field_name].tolist()
        
        field_analysis = {
            'field_name': field_name,
            'total_samples': len(predicted),
            'error_breakdown': defaultdict(int),
            'severity_scores': [],
            'similarity_scores': [],
            'error_details': [],
            'worst_errors': [],
            'best_partial_matches': [],
            'accuracy_metrics': {}
        }
        
        # Analyze each sample
        for i, (pred, act) in enumerate(zip(predicted, actual)):
            error_info = self.categorize_error(pred, act, field_name)
            
            field_analysis['error_breakdown'][error_info['error_type']] += 1
            field_analysis['severity_scores'].append(error_info['severity'])
            field_analysis['similarity_scores'].append(error_info['similarity'])
            
            error_detail = {
                'sample_index': i,
                'error_type': error_info['error_type'],
                'severity': error_info['severity'],
                'similarity': error_info['similarity'],
                'predicted': pred,
                'actual': act,
                'details': error_info['details']
            }
            
            field_analysis['error_details'].append(error_detail)
            
            # Track worst errors
            if error_info['severity'] >= 0.8:
                field_analysis['worst_errors'].append(error_detail)
            
            # Track good partial matches for improvement insights
            elif error_info['error_type'] == 'partial' and error_info['similarity'] >= 0.6:
                field_analysis['best_partial_matches'].append(error_detail)
        
        # Calculate summary metrics
        total_samples = field_analysis['total_samples']
        correct_count = field_analysis['error_breakdown']['none']
        
        field_analysis['accuracy_metrics'] = {
            'exact_accuracy': correct_count / total_samples if total_samples > 0 else 0,
            'avg_severity': np.mean(field_analysis['severity_scores']),
            'avg_similarity': np.mean(field_analysis['similarity_scores']),
            'error_rate': (total_samples - correct_count) / total_samples if total_samples > 0 else 0
        }
        
        # Sort errors by severity for reporting
        field_analysis['worst_errors'] = sorted(
            field_analysis['worst_errors'], 
            key=lambda x: x['severity'], 
            reverse=True
        )[:5]
        
        field_analysis['best_partial_matches'] = sorted(
            field_analysis['best_partial_matches'],
            key=lambda x: x['similarity'],
            reverse=True
        )[:3]
        
        return field_analysis
    
    def analyze_medical_specialty_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns by medical specialty."""
        if 'medical_specialty' not in self.results_df.columns:
            return {'error': 'Medical specialty information not available'}
        
        specialty_analysis = defaultdict(lambda: {
            'total_samples': 0,
            'field_errors': defaultdict(list),
            'overall_accuracy': 0,
            'error_types': defaultdict(int)
        })
        
        # Analyze by specialty
        for idx, specialty in enumerate(self.results_df['medical_specialty']):
            specialty_analysis[specialty]['total_samples'] += 1
            
            specialty_errors = 0
            total_fields = 0
            
            for field in self.common_fields:
                if field == 'medical_specialty':
                    continue
                
                predicted = self.results_df.iloc[idx][field]
                actual = self.ground_truth_df.iloc[idx][field] if idx < len(self.ground_truth_df) else None
                
                if actual is not None:
                    error_info = self.categorize_error(predicted, actual, field)
                    
                    specialty_analysis[specialty]['field_errors'][field].append({
                        'sample_index': idx,
                        'error_type': error_info['error_type'],
                        'severity': error_info['severity'],
                        'similarity': error_info['similarity']
                    })
                    
                    specialty_analysis[specialty]['error_types'][error_info['error_type']] += 1
                    
                    if error_info['error_type'] != 'none':
                        specialty_errors += 1
                    
                    total_fields += 1
            
            # Calculate accuracy for this sample
            if total_fields > 0:
                sample_accuracy = (total_fields - specialty_errors) / total_fields
                current_total = specialty_analysis[specialty]['total_samples']
                current_avg = specialty_analysis[specialty]['overall_accuracy']
                
                # Update running average
                specialty_analysis[specialty]['overall_accuracy'] = (
                    (current_avg * (current_total - 1) + sample_accuracy) / current_total
                )
        
        return dict(specialty_analysis)
    
    def analyze_text_complexity_patterns(self, transcriptions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze error patterns based on text complexity."""
        if transcriptions is None:
            return {'error': 'Original transcriptions not provided'}
        
        complexity_analysis = []
        
        for idx, transcription in enumerate(transcriptions):
            if idx >= len(self.results_df):
                break
            
            text = str(transcription).lower()
            
            # Calculate complexity metrics
            complexity_metrics = {
                'sample_index': idx,
                'text_length': len(text),
                'word_count': len(text.split()),
                'sentence_count': len([s for s in text.split('.') if s.strip()]),
                'complexity_indicators': sum(1 for indicator in self.complexity_indicators if indicator in text),
                'medical_terms': len(re.findall(r'\b(?:diagnosis|treatment|medication|procedure|syndrome|disease)\b', text)),
                'numbers_count': len(re.findall(r'\d+', text))
            }
            
            # Calculate error rate for this sample
            sample_errors = 0
            total_fields = 0
            
            for field in self.common_fields:
                if field == 'medical_specialty':
                    continue
                
                predicted = self.results_df.iloc[idx][field]
                actual = self.ground_truth_df.iloc[idx][field] if idx < len(self.ground_truth_df) else None
                
                if actual is not None:
                    error_info = self.categorize_error(predicted, actual, field)
                    if error_info['error_type'] != 'none':
                        sample_errors += 1
                    total_fields += 1
            
            complexity_metrics['error_rate'] = sample_errors / total_fields if total_fields > 0 else 0
            complexity_metrics['total_errors'] = sample_errors
            
            complexity_analysis.append(complexity_metrics)
        
        return complexity_analysis
    
    def compare_extractors(self, baseline_results: pd.DataFrame, openai_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare baseline and OpenAI extractors to find where each performs better.
        """
        comparison = {
            'baseline_better': [],
            'openai_better': [],
            'both_wrong': [],
            'both_correct': [],
            'field_performance': {},
            'overall_comparison': {}
        }
        
        # Ensure consistent length
        min_len = min(len(baseline_results), len(openai_results), len(self.ground_truth_df))
        baseline_results = baseline_results.iloc[:min_len]
        openai_results = openai_results.iloc[:min_len]
        ground_truth = self.ground_truth_df.iloc[:min_len]
        
        for field in self.common_fields:
            if field == 'medical_specialty':
                continue
            
            if field not in baseline_results.columns or field not in openai_results.columns:
                continue
            
            comparison['field_performance'][field] = {
                'baseline_better_count': 0,
                'openai_better_count': 0,
                'tie_count': 0,
                'both_wrong_count': 0,
                'both_correct_count': 0,
                'examples': []
            }
            
            for i in range(min_len):
                actual = ground_truth.iloc[i][field]
                baseline_pred = baseline_results.iloc[i][field]
                openai_pred = openai_results.iloc[i][field]
                
                # Calculate similarities
                baseline_error = self.categorize_error(baseline_pred, actual, field)
                openai_error = self.categorize_error(openai_pred, actual, field)
                
                baseline_score = 1 - baseline_error['severity']
                openai_score = 1 - openai_error['severity']
                
                example = {
                    'sample_index': i,
                    'field': field,
                    'actual': actual,
                    'baseline_prediction': baseline_pred,
                    'openai_prediction': openai_pred,
                    'baseline_score': baseline_score,
                    'openai_score': openai_score,
                    'baseline_error_type': baseline_error['error_type'],
                    'openai_error_type': openai_error['error_type']
                }
                
                # Categorize performance
                if baseline_score > openai_score + 0.1:  # Baseline significantly better
                    comparison['baseline_better'].append(example)
                    comparison['field_performance'][field]['baseline_better_count'] += 1
                elif openai_score > baseline_score + 0.1:  # OpenAI significantly better
                    comparison['openai_better'].append(example)
                    comparison['field_performance'][field]['openai_better_count'] += 1
                else:  # Tie
                    comparison['field_performance'][field]['tie_count'] += 1
                    
                    if baseline_score < 0.3 and openai_score < 0.3:  # Both wrong
                        comparison['both_wrong'].append(example)
                        comparison['field_performance'][field]['both_wrong_count'] += 1
                    elif baseline_score > 0.8 and openai_score > 0.8:  # Both correct
                        comparison['both_correct'].append(example)
                        comparison['field_performance'][field]['both_correct_count'] += 1
                
                # Store examples for detailed analysis
                if len(comparison['field_performance'][field]['examples']) < 3:
                    comparison['field_performance'][field]['examples'].append(example)
        
        # Calculate overall comparison metrics
        total_baseline_better = len(comparison['baseline_better'])
        total_openai_better = len(comparison['openai_better'])
        total_comparisons = total_baseline_better + total_openai_better + comparison['field_performance'].get('tie_count', 0)
        
        comparison['overall_comparison'] = {
            'baseline_win_rate': total_baseline_better / max(total_comparisons, 1),
            'openai_win_rate': total_openai_better / max(total_comparisons, 1),
            'total_comparisons': total_comparisons,
            'improvement_areas': self._identify_improvement_areas(comparison)
        }
        
        return comparison
    
    def _identify_improvement_areas(self, comparison: Dict) -> List[Dict]:
        """Identify specific areas for improvement based on comparison."""
        improvements = []
        
        for field, perf in comparison['field_performance'].items():
            total_field_comparisons = (perf['baseline_better_count'] + 
                                     perf['openai_better_count'] + 
                                     perf['tie_count'])
            
            if total_field_comparisons == 0:
                continue
            
            baseline_rate = perf['baseline_better_count'] / total_field_comparisons
            openai_rate = perf['openai_better_count'] / total_field_comparisons
            both_wrong_rate = perf['both_wrong_count'] / total_field_comparisons
            
            if both_wrong_rate > 0.3:
                improvements.append({
                    'field': field,
                    'issue': 'high_failure_rate',
                    'description': f'Both extractors fail frequently ({both_wrong_rate:.1%})',
                    'recommendation': 'Fundamental approach review needed'
                })
            elif baseline_rate > openai_rate + 0.2:
                improvements.append({
                    'field': field,
                    'issue': 'openai_underperforming',
                    'description': f'Baseline outperforms OpenAI ({baseline_rate:.1%} vs {openai_rate:.1%})',
                    'recommendation': 'Improve OpenAI prompts or use hybrid approach'
                })
            elif openai_rate > baseline_rate + 0.2:
                improvements.append({
                    'field': field,
                    'issue': 'baseline_underperforming',
                    'description': f'OpenAI outperforms baseline ({openai_rate:.1%} vs {baseline_rate:.1%})',
                    'recommendation': 'Improve regex patterns or switch to OpenAI'
                })
        
        return improvements
    
    def generate_actionable_insights(self) -> List[Dict[str, str]]:
        """Generate specific, actionable insights for improvement."""
        insights = []
        
        if not self.field_errors:
            insights.append({
                'type': 'error',
                'message': 'Run complete analysis first using run_complete_analysis()',
                'action': 'Execute analysis'
            })
            return insights
        
        # Overall performance insights
        overall_accuracy = np.mean([
            analysis['accuracy_metrics']['exact_accuracy'] 
            for analysis in self.field_errors.values()
        ])
        
        if overall_accuracy < 0.6:
            insights.append({
                'type': 'critical',
                'message': f'Overall accuracy critically low ({overall_accuracy:.1%})',
                'action': 'Complete methodology review required'
            })
        elif overall_accuracy < 0.8:
            insights.append({
                'type': 'warning',
                'message': f'Overall accuracy below target ({overall_accuracy:.1%})',
                'action': 'Focus on top error patterns'
            })
        else:
            insights.append({
                'type': 'success',
                'message': f'Good overall accuracy ({overall_accuracy:.1%})',
                'action': 'Fine-tune for optimal performance'
            })
        
        # Field-specific insights
        for field, analysis in self.field_errors.items():
            field_accuracy = analysis['accuracy_metrics']['exact_accuracy']
            error_breakdown = analysis['error_breakdown']
            total_errors = sum(error_breakdown.values()) - error_breakdown['none']
            
            if field_accuracy < 0.5:
                insights.append({
                    'type': 'critical',
                    'message': f'{field} field severely underperforming ({field_accuracy:.1%})',
                    'action': f'Redesign {field} extraction completely'
                })
            
            # Specific error type insights
            if total_errors > 0:
                missing_rate = error_breakdown['missing'] / total_errors
                incorrect_rate = error_breakdown['incorrect'] / total_errors
                partial_rate = error_breakdown['partial'] / total_errors
                
                if missing_rate > 0.5:
                    insights.append({
                        'type': 'opportunity',
                        'message': f'{field}: High missing data rate ({missing_rate:.1%})',
                        'action': 'Improve pattern detection or add fallback methods'
                    })
                
                if incorrect_rate > 0.3:
                    insights.append({
                        'type': 'warning',
                        'message': f'{field}: High incorrect extraction rate ({incorrect_rate:.1%})',
                        'action': 'Review and refine extraction logic'
                    })
                
                if partial_rate > 0.4:
                    insights.append({
                        'type': 'opportunity',
                        'message': f'{field}: Many partial matches ({partial_rate:.1%})',
                        'action': 'Fine-tune to capture complete information'
                    })
        
        # Medical specialty insights
        if 'medical_specialty_patterns' in self.pattern_analysis:
            specialty_errors = self.pattern_analysis['medical_specialty_patterns']
            if specialty_errors and 'error' not in specialty_errors:
                worst_specialty = min(specialty_errors.items(), 
                                    key=lambda x: x[1]['overall_accuracy'])
                insights.append({
                    'type': 'focus',
                    'message': f'{worst_specialty[0]} specialty has lowest accuracy ({worst_specialty[1]["overall_accuracy"]:.1%})',
                    'action': f'Develop {worst_specialty[0]}-specific extraction rules'
                })
        
        return insights
    
    def create_visualizations(self, save_plots: bool = True, show_plots: bool = True) -> Dict[str, Any]:
        """Create comprehensive error analysis visualizations."""
        try:
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
            fig = plt.figure(figsize=(20, 15))
            
            # Create grid layout
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Field Accuracy Comparison
            ax1 = fig.add_subplot(gs[0, 0])
            if self.field_errors:
                fields = list(self.field_errors.keys())
                accuracies = [self.field_errors[field]['accuracy_metrics']['exact_accuracy'] 
                             for field in fields]
                
                colors = ['#e74c3c' if acc < 0.7 else '#f39c12' if acc < 0.85 else '#27ae60' 
                         for acc in accuracies]
                
                bars = ax1.bar(range(len(fields)), accuracies, color=colors)
                ax1.set_title('Field Accuracy Comparison', fontweight='bold')
                ax1.set_ylabel('Accuracy')
                ax1.set_xticks(range(len(fields)))
                ax1.set_xticklabels([f.replace('_', '\n') for f in fields], rotation=0, ha='center')
                ax1.set_ylim(0, 1)
                
                # Add accuracy labels
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
            
            # 2. Error Type Distribution
            ax2 = fig.add_subplot(gs[0, 1])
            if self.field_errors:
                all_errors = defaultdict(int)
                for analysis in self.field_errors.values():
                    for error_type, count in analysis['error_breakdown'].items():
                        if error_type != 'none':
                            all_errors[error_type] += count
                
                if all_errors:
                    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#e67e22']
                    wedges, texts, autotexts = ax2.pie(
                        all_errors.values(), 
                        labels=all_errors.keys(),
                        autopct='%1.1f%%',
                        colors=colors[:len(all_errors)],
                        startangle=90
                    )
                    ax2.set_title('Error Type Distribution', fontweight='bold')
            
            # 3. Severity Heatmap by Field
            ax3 = fig.add_subplot(gs[0, 2])
            if self.field_errors:
                severity_matrix = []
                field_names = []
                
                for field, analysis in self.field_errors.items():
                    severity_by_type = [0] * 5  # missing, incorrect, partial, overextracted, format_error
                    error_types = ['missing', 'incorrect', 'partial', 'overextracted', 'format_error']
                    
                    for i, error_type in enumerate(error_types):
                        if error_type in analysis['error_breakdown']:
                            severity_by_type[i] = analysis['error_breakdown'][error_type]
                    
                    severity_matrix.append(severity_by_type)
                    field_names.append(field.replace('_', '\n'))
                
                if severity_matrix:
                    im = ax3.imshow(severity_matrix, cmap='Reds', aspect='auto')
                    ax3.set_xticks(range(len(error_types)))
                    ax3.set_xticklabels([et.replace('_', '\n') for et in error_types], rotation=45)
                    ax3.set_yticks(range(len(field_names)))
                    ax3.set_yticklabels(field_names)
                    ax3.set_title('Error Count Heatmap', fontweight='bold')
                    
                    # Add text annotations
                    for i in range(len(field_names)):
                        for j in range(len(error_types)):
                            text = ax3.text(j, i, severity_matrix[i][j],
                                          ha="center", va="center", color="black" if severity_matrix[i][j] < 2 else "white")
            
            # 4. Medical Specialty Performance (if available)
            ax4 = fig.add_subplot(gs[1, :2])
            if ('medical_specialty_patterns' in self.pattern_analysis and 
                'error' not in self.pattern_analysis['medical_specialty_patterns']):
                
                specialty_data = self.pattern_analysis['medical_specialty_patterns']
                specialties = list(specialty_data.keys())
                accuracies = [specialty_data[spec]['overall_accuracy'] for spec in specialties]
                
                bars = ax4.bar(specialties, accuracies, color='lightblue', edgecolor='navy')
                ax4.set_title('Accuracy by Medical Specialty', fontweight='bold')
                ax4.set_ylabel('Accuracy')
                ax4.set_ylim(0, 1)
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
                
                # Add accuracy labels
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{acc:.1%}', ha='center', va='bottom')
            
            # 5. Text Complexity vs Error Rate (if available)
            ax5 = fig.add_subplot(gs[1, 2])
            if ('text_complexity_patterns' in self.pattern_analysis and 
                'error' not in self.pattern_analysis['text_complexity_patterns']):
                
                complexity_data = self.pattern_analysis['text_complexity_patterns']
                text_lengths = [d['text_length'] for d in complexity_data]
                error_rates = [d['error_rate'] for d in complexity_data]
                
                ax5.scatter(text_lengths, error_rates, alpha=0.6, color='coral')
                ax5.set_xlabel('Text Length (characters)')
                ax5.set_ylabel('Error Rate')
                ax5.set_title('Text Complexity vs Error Rate', fontweight='bold')
                
                # Add trend line
                if len(text_lengths) > 1:
                    z = np.polyfit(text_lengths, error_rates, 1)
                    p = np.poly1d(z)
                    ax5.plot(text_lengths, p(text_lengths), "r--", alpha=0.8)
            
            # 6. Error Severity Distribution
            ax6 = fig.add_subplot(gs[2, 0])
            if self.field_errors:
                all_severities = []
                for analysis in self.field_errors.values():
                    all_severities.extend(analysis['severity_scores'])
                
                if all_severities:
                    ax6.hist(all_severities, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
                    ax6.set_xlabel('Error Severity')
                    ax6.set_ylabel('Frequency')
                    ax6.set_title('Error Severity Distribution', fontweight='bold')
                    ax6.axvline(np.mean(all_severities), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(all_severities):.2f}')
                    ax6.legend()
            
            # 7. Top Error Examples
            ax7 = fig.add_subplot(gs[2, 1:])
            ax7.axis('off')
            
            # Create text summary of top errors
            error_text = "TOP ERROR EXAMPLES:\n" + "="*50 + "\n"
            
            for field, analysis in list(self.field_errors.items())[:3]:
                if analysis['worst_errors']:
                    error_text += f"\n{field.upper()}:\n"
                    for i, error in enumerate(analysis['worst_errors'][:2]):
                        error_text += f"{i+1}. {error['details'][:80]}...\n"
            
            ax7.text(0.05, 0.95, error_text, transform=ax7.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.suptitle('Medical Transcription Error Analysis Dashboard', 
                        fontsize=16, fontweight='bold')
            
            if save_plots:
                filename = 'comprehensive_error_analysis.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"📊 Visualizations saved to '{filename}'")
            
            if show_plots:
                plt.show()
            else:
                plt.close()
            
            return {
                'visualization_created': True,
                'filename': filename if save_plots else None,
                'charts_created': 7
            }
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            print("💡 Try installing required packages: pip install matplotlib seaborn")
            return {
                'visualization_created': False,
                'error': str(e)
            }
    
    def run_complete_analysis(self, transcriptions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive error analysis across all dimensions."""
        print("🔍 Running comprehensive error analysis...")
        
        # Analyze each field
        print("📊 Analyzing field-level errors...")
        for field in self.common_fields:
            if field != 'medical_specialty':
                print(f"  - {field}")
                self.field_errors[field] = self.analyze_field_errors(field)
        
        # Analyze patterns
        print("🏥 Analyzing medical specialty patterns...")
        self.pattern_analysis['medical_specialty_patterns'] = self.analyze_medical_specialty_patterns()
        
        if transcriptions:
            print("📝 Analyzing text complexity patterns...")
            self.pattern_analysis['text_complexity_patterns'] = self.analyze_text_complexity_patterns(transcriptions)
        
        # Compile complete analysis
        self.error_analysis = {
            'field_errors': self.field_errors,
            'pattern_analysis': self.pattern_analysis,
            'summary_statistics': self._generate_summary_statistics(),
            'actionable_insights': self.generate_actionable_insights()
        }
        
        print("✅ Error analysis complete!")
        return self.error_analysis
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        if not self.field_errors:
            return {}
        
        # Field-level statistics
        field_accuracies = {}
        field_error_counts = {}
        all_error_types = defaultdict(int)
        
        for field, analysis in self.field_errors.items():
            field_accuracies[field] = analysis['accuracy_metrics']['exact_accuracy']
            field_error_counts[field] = len(analysis['error_details'])
            
            for error_type, count in analysis['error_breakdown'].items():
                all_error_types[error_type] += count
        
        # Overall statistics
        overall_accuracy = np.mean(list(field_accuracies.values()))
        best_field = max(field_accuracies, key=field_accuracies.get) if field_accuracies else ""
        worst_field = min(field_accuracies, key=field_accuracies.get) if field_accuracies else ""
        
        summary = {
            'total_fields_analyzed': len(self.field_errors),
            'total_samples': len(self.results_df),
            'overall_accuracy': overall_accuracy,
            'best_performing_field': best_field,
            'worst_performing_field': worst_field,
            'field_performance': {
                field: {
                    'accuracy': field_accuracies[field],
                    'error_count': field_error_counts[field],
                    'avg_severity': self.field_errors[field]['accuracy_metrics']['avg_severity']
                }
                for field in field_accuracies
            },
            'error_type_distribution': dict(all_error_types),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate specific recommendations based on analysis."""
        recommendations = []
        
        if not self.field_errors:
            return ["Complete field analysis first"]
        
        # Overall performance recommendations
        overall_accuracy = np.mean([
            analysis['accuracy_metrics']['exact_accuracy'] 
            for analysis in self.field_errors.values()
        ])
        
        if overall_accuracy < 0.7:
            recommendations.append("🚨 URGENT: Overall accuracy below 70% - fundamental review needed")
        elif overall_accuracy < 0.85:
            recommendations.append("⚠️ PRIORITY: Overall accuracy below 85% - systematic improvements required")
        
        # Field-specific recommendations
        for field, analysis in self.field_errors.items():
            accuracy = analysis['accuracy_metrics']['exact_accuracy']
            error_breakdown = analysis['error_breakdown']
            
            if accuracy < 0.6:
                recommendations.append(f"🎯 CRITICAL: {field} accuracy critically low ({accuracy:.1%}) - redesign extraction")
            
            # Error type specific recommendations
            total_errors = sum(v for k, v in error_breakdown.items() if k != 'none')
            if total_errors > 0:
                missing_rate = error_breakdown.get('missing', 0) / total_errors
                incorrect_rate = error_breakdown.get('incorrect', 0) / total_errors
                
                if missing_rate > 0.4:
                    recommendations.append(f"📝 {field}: High missing rate ({missing_rate:.1%}) - improve detection patterns")
                
                if incorrect_rate > 0.3:
                    recommendations.append(f"🔧 {field}: High incorrect rate ({incorrect_rate:.1%}) - refine extraction logic")
        
        # Medical specialty recommendations
        if ('medical_specialty_patterns' in self.pattern_analysis and 
            'error' not in self.pattern_analysis['medical_specialty_patterns']):
            
            specialty_data = self.pattern_analysis['medical_specialty_patterns']
            worst_specialty = min(specialty_data.items(), key=lambda x: x[1]['overall_accuracy'])
            
            if worst_specialty[1]['overall_accuracy'] < 0.7:
                recommendations.append(f"🏥 SPECIALTY FOCUS: {worst_specialty[0]} needs specialized extraction rules")
        
        return recommendations
    
    def generate_detailed_report(self, filename: str = "comprehensive_error_analysis.txt") -> str:
        """Generate a comprehensive, detailed error analysis report."""
        if not self.error_analysis:
            return "Run complete analysis first using run_complete_analysis()"
        
        report_lines = []
        
        # Header
        report_lines.extend([
            "=" * 100,
            "COMPREHENSIVE MEDICAL TRANSCRIPTION ERROR ANALYSIS REPORT",
            "=" * 100,
            "",
            f"Generated for {len(self.results_df)} medical transcription samples",
            f"Analyzed {len(self.field_errors)} fields: {', '.join(self.field_errors.keys())}",
            ""
        ])
        
        # Executive Summary
        summary = self.error_analysis['summary_statistics']
        report_lines.extend([
            "EXECUTIVE SUMMARY",
            "-" * 50,
            f"Overall Accuracy: {summary['overall_accuracy']:.1%}",
            f"Best Performing Field: {summary['best_performing_field']} ({summary['field_performance'][summary['best_performing_field']]['accuracy']:.1%})",
            f"Worst Performing Field: {summary['worst_performing_field']} ({summary['field_performance'][summary['worst_performing_field']]['accuracy']:.1%})",
            f"Total Error Instances: {sum(summary['error_type_distribution'].values()) - summary['error_type_distribution'].get('none', 0)}",
            ""
        ])
        
        # Field-by-Field Analysis
        report_lines.extend([
            "DETAILED FIELD ANALYSIS",
            "-" * 50
        ])
        
        for field, analysis in self.field_errors.items():
            metrics = analysis['accuracy_metrics']
            report_lines.extend([
                f"\n{field.upper().replace('_', ' ')}:",
                f"  Accuracy: {metrics['exact_accuracy']:.1%}",
                f"  Average Severity: {metrics['avg_severity']:.2f}",
                f"  Error Rate: {metrics['error_rate']:.1%}",
                f"  Total Errors: {len(analysis['error_details'])}",
                ""
            ])
            
            # Error type breakdown
            report_lines.append("  Error Type Breakdown:")
            for error_type, count in analysis['error_breakdown'].items():
                percentage = (count / analysis['total_samples']) * 100
                report_lines.append(f"    {error_type.title()}: {count} ({percentage:.1f}%)")
            
            # Worst errors examples
            if analysis['worst_errors']:
                report_lines.append("\n  Critical Error Examples:")
                for i, error in enumerate(analysis['worst_errors'][:3]):
                    report_lines.append(f"    {i+1}. Sample {error['sample_index']}: {error['details']}")
            
            report_lines.append("")
        
        # Error Pattern Analysis
        if 'medical_specialty_patterns' in self.pattern_analysis:
            specialty_data = self.pattern_analysis['medical_specialty_patterns']
            if 'error' not in specialty_data:
                report_lines.extend([
                    "MEDICAL SPECIALTY ANALYSIS",
                    "-" * 50
                ])
                
                for specialty, data in specialty_data.items():
                    report_lines.extend([
                        f"\n{specialty}:",
                        f"  Samples: {data['total_samples']}",
                        f"  Accuracy: {data['overall_accuracy']:.1%}",
                        f"  Common Error Types: {', '.join([k for k, v in data['error_types'].items() if v > 0 and k != 'none'])}"
                    ])
                report_lines.append("")
        
        # Text Complexity Analysis
        if 'text_complexity_patterns' in self.pattern_analysis:
            complexity_data = self.pattern_analysis['text_complexity_patterns']
            if 'error' not in complexity_data:
                avg_length = np.mean([d['text_length'] for d in complexity_data])
                avg_error_rate = np.mean([d['error_rate'] for d in complexity_data])
                
                report_lines.extend([
                    "TEXT COMPLEXITY ANALYSIS",
                    "-" * 50,
                    f"Average Text Length: {avg_length:.0f} characters",
                    f"Average Error Rate: {avg_error_rate:.1%}",
                    f"Complexity Correlation: {'High' if avg_error_rate > 0.3 else 'Moderate' if avg_error_rate > 0.15 else 'Low'}",
                    ""
                ])
        
        # Actionable Insights
        insights = self.error_analysis['actionable_insights']
        report_lines.extend([
            "ACTIONABLE INSIGHTS & RECOMMENDATIONS",
            "-" * 50
        ])
        
        for insight in insights:
            icon = "🚨" if insight['type'] == 'critical' else "⚠️" if insight['type'] == 'warning' else "💡"
            report_lines.append(f"{icon} {insight['message']}")
            report_lines.append(f"   ACTION: {insight['action']}")
            report_lines.append("")
        
        # Recommendations
        recommendations = summary['recommendations']
        report_lines.extend([
            "PRIORITIZED RECOMMENDATIONS",
            "-" * 50
        ])
        
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"{i}. {rec}")
        
        report_lines.extend([
            "",
            "=" * 100,
            "END OF REPORT",
            "=" * 100
        ])
        
        # Write to file
        report_text = "\n".join(report_lines)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Comprehensive error analysis report saved to '{filename}'")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
        
        return report_text

    def export_error_data(self, filename: str = "error_analysis_data.csv") -> bool:
        """Export detailed error data to CSV for further analysis."""
        try:
            all_error_records = []
            
            for field, analysis in self.field_errors.items():
                for error_detail in analysis['error_details']:
                    record = {
                        'field': field,
                        'sample_index': error_detail['sample_index'],
                        'error_type': error_detail['error_type'],
                        'severity': error_detail['severity'],
                        'similarity': error_detail['similarity'],
                        'predicted_value': error_detail['predicted'],
                        'actual_value': error_detail['actual'],
                        'error_description': error_detail['details']
                    }
                    
                    # Add medical specialty if available
                    if ('medical_specialty' in self.results_df.columns and 
                        error_detail['sample_index'] < len(self.results_df)):
                        record['medical_specialty'] = self.results_df.iloc[error_detail['sample_index']]['medical_specialty']
                    
                    all_error_records.append(record)
            
            error_df = pd.DataFrame(all_error_records)
            error_df.to_csv(filename, index=False)
            
            print(f"📊 Error data exported to '{filename}' ({len(all_error_records)} error records)")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            return False