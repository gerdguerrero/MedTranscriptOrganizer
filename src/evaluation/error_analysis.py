import pandas as pd
from typing import Dict, List, Any, Tuple
import re
from collections import Counter

class ErrorAnalysis:
    """
    Error analysis functionality for medical information extraction.
    Provides detailed analysis of extraction errors and failure modes.
    """
    
    def __init__(self):
        self.error_categories = {
            "missing_information": "Information not present in transcription",
            "incorrect_extraction": "Wrong information extracted",
            "partial_extraction": "Only part of the information extracted",
            "format_error": "Correct information but wrong format",
            "confidence_miscalibration": "Confidence score doesn't match accuracy"
        }
    
    def analyze_errors(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame, 
                      transcriptions: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive error analysis.
        
        Args:
            predictions: Extracted information
            ground_truth: Correct annotations
            transcriptions: Original transcription texts
            
        Returns:
            Detailed error analysis results
        """
        
        analysis = {
            "error_breakdown": {},
            "common_failure_modes": {},
            "confidence_analysis": {},
            "field_specific_errors": {}
        }
        
        fields = ["age", "recommended_treatment", "primary_diagnosis", "icd_10_code"]
        
        for field in fields:
            if field in predictions.columns and field in ground_truth.columns:
                field_analysis = self._analyze_field_errors(
                    predictions[field].tolist(),
                    ground_truth[field].tolist(),
                    transcriptions,
                    field
                )
                analysis["field_specific_errors"][field] = field_analysis
        
        # Overall error breakdown
        analysis["error_breakdown"] = self._calculate_error_breakdown(analysis["field_specific_errors"])
        
        # Confidence analysis
        if "overall_confidence" in predictions.columns:
            analysis["confidence_analysis"] = self._analyze_confidence_errors(
                predictions, ground_truth
            )
        
        # Common failure modes
        analysis["common_failure_modes"] = self._identify_failure_modes(
            predictions, ground_truth, transcriptions
        )
        
        return analysis
    
    def _analyze_field_errors(self, predictions: List[str], ground_truth: List[str], 
                             transcriptions: List[str], field: str) -> Dict[str, Any]:
        """Analyze errors for a specific field."""
        
        errors = []
        error_types = []
        
        for i, (pred, gt, transcript) in enumerate(zip(predictions, ground_truth, transcriptions)):
            pred = pred.strip()
            gt = gt.strip()
            
            if pred.lower() == gt.lower():
                continue  # Correct prediction
            
            error_info = {
                "index": i,
                "predicted": pred,
                "ground_truth": gt,
                "transcription_snippet": transcript[:200] + "..." if len(transcript) > 200 else transcript
            }
            
            # Categorize error type
            error_type = self._categorize_error(pred, gt, transcript, field)
            error_info["error_type"] = error_type
            
            errors.append(error_info)
            error_types.append(error_type)
        
        # Error statistics
        error_counts = Counter(error_types)
        total_errors = len(errors)
        total_predictions = len(predictions)
        
        return {
            "errors": errors,
            "error_counts": dict(error_counts),
            "error_rate": total_errors / total_predictions,
            "most_common_error": error_counts.most_common(1)[0] if error_counts else None
        }
    
    def _categorize_error(self, prediction: str, ground_truth: str, 
                         transcription: str, field: str) -> str:
        """Categorize the type of error made."""
        
        pred_lower = prediction.lower()
        gt_lower = ground_truth.lower()
        transcript_lower = transcription.lower()
        
        # Check if ground truth information is present in transcription
        gt_words = set(re.findall(r'\b\w+\b', gt_lower))
        transcript_words = set(re.findall(r'\b\w+\b', transcript_lower))
        
        overlap = len(gt_words & transcript_words) / len(gt_words) if gt_words else 0
        
        if overlap < 0.3:  # Ground truth not well represented in transcript
            return "missing_information"
        
        # Check for partial extraction
        pred_words = set(re.findall(r'\b\w+\b', pred_lower))
        pred_gt_overlap = len(pred_words & gt_words) / len(gt_words) if gt_words else 0
        
        if 0.3 <= pred_gt_overlap < 0.8:
            return "partial_extraction"
        
        # Check for format errors (especially for ICD-10 codes)
        if field == "icd_10_code":
            if self._is_format_error(prediction, ground_truth):
                return "format_error"
        
        # Default to incorrect extraction
        return "incorrect_extraction"
    
    def _is_format_error(self, prediction: str, ground_truth: str) -> bool:
        """Check if error is due to format differences (e.g., ICD-10 codes)."""
        # Remove common formatting differences
        pred_clean = re.sub(r'[.\-\s]', '', prediction.upper())
        gt_clean = re.sub(r'[.\-\s]', '', ground_truth.upper())
        
        return pred_clean == gt_clean
    
    def _calculate_error_breakdown(self, field_errors: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate overall error breakdown across all fields."""
        
        total_error_counts = Counter()
        total_errors = 0
        
        for field_data in field_errors.values():
            for error_type, count in field_data["error_counts"].items():
                total_error_counts[error_type] += count
                total_errors += count
        
        if total_errors == 0:
            return {}
        
        return {error_type: count / total_errors 
                for error_type, count in total_error_counts.items()}
    
    def _analyze_confidence_errors(self, predictions: pd.DataFrame, 
                                  ground_truth: pd.DataFrame) -> Dict[str, Any]:
        """Analyze confidence calibration errors."""
        
        confidence_analysis = {
            "overconfident_errors": [],
            "underconfident_correct": [],
            "confidence_accuracy_correlation": {}
        }
        
        fields = ["age", "recommended_treatment", "primary_diagnosis", "icd_10_code"]
        
        for field in fields:
            if field in predictions.columns and field in ground_truth.columns:
                pred_values = predictions[field].tolist()
                gt_values = ground_truth[field].tolist()
                
                if "overall_confidence" in predictions.columns:
                    confidences = predictions["overall_confidence"].tolist()
                    
                    for i, (pred, gt, conf) in enumerate(zip(pred_values, gt_values, confidences)):
                        is_correct = pred.strip().lower() == gt.strip().lower()
                        
                        # High confidence but wrong
                        if conf > 0.8 and not is_correct:
                            confidence_analysis["overconfident_errors"].append({
                                "index": i,
                                "field": field,
                                "confidence": conf,
                                "predicted": pred,
                                "ground_truth": gt
                            })
                        
                        # Low confidence but correct
                        if conf < 0.3 and is_correct:
                            confidence_analysis["underconfident_correct"].append({
                                "index": i,
                                "field": field,
                                "confidence": conf,
                                "predicted": pred,
                                "ground_truth": gt
                            })
        
        return confidence_analysis
    
    def _identify_failure_modes(self, predictions: pd.DataFrame, ground_truth: pd.DataFrame,
                               transcriptions: List[str]) -> Dict[str, List[Dict]]:
        """Identify common failure modes across the dataset."""
        
        failure_modes = {
            "systematic_errors": [],
            "transcription_quality_issues": [],
            "complex_cases": []
        }
        
        # Analyze transcription lengths and complexity
        for i, transcript in enumerate(transcriptions):
            transcript_length = len(transcript.split())
            
            # Very short transcriptions
            if transcript_length < 50:
                failure_modes["transcription_quality_issues"].append({
                    "index": i,
                    "issue": "very_short_transcript",
                    "length": transcript_length,
                    "snippet": transcript[:100]
                })
            
            # Very long transcriptions
            elif transcript_length > 1000:
                failure_modes["complex_cases"].append({
                    "index": i,
                    "issue": "very_long_transcript",
                    "length": transcript_length,
                    "snippet": transcript[:100] + "..."
                })
        
        return failure_modes
    
    def generate_error_report(self, error_analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive error analysis report."""
        
        report = "Error Analysis Report\n"
        report += "=" * 40 + "\n\n"
        
        # Overall error breakdown
        if "error_breakdown" in error_analysis:
            report += "OVERALL ERROR BREAKDOWN\n"
            report += "-" * 25 + "\n"
            
            for error_type, percentage in error_analysis["error_breakdown"].items():
                report += f"{error_type}: {percentage:.1%}\n"
            report += "\n"
        
        # Field-specific errors
        if "field_specific_errors" in error_analysis:
            report += "FIELD-SPECIFIC ERROR ANALYSIS\n"
            report += "-" * 35 + "\n"
            
            for field, field_data in error_analysis["field_specific_errors"].items():
                report += f"\n{field.upper()}:\n"
                report += f"  Error rate: {field_data['error_rate']:.1%}\n"
                
                if field_data["most_common_error"]:
                    error_type, count = field_data["most_common_error"]
                    report += f"  Most common error: {error_type} ({count} cases)\n"
        
        # Confidence analysis
        if "confidence_analysis" in error_analysis:
            conf_analysis = error_analysis["confidence_analysis"]
            
            report += "\nCONFIDENCE ANALYSIS\n"
            report += "-" * 20 + "\n"
            
            overconfident = len(conf_analysis["overconfident_errors"])
            underconfident = len(conf_analysis["underconfident_correct"])
            
            report += f"Overconfident errors: {overconfident}\n"
            report += f"Underconfident correct predictions: {underconfident}\n"
        
        # Common failure modes
        if "common_failure_modes" in error_analysis:
            failure_modes = error_analysis["common_failure_modes"]
            
            report += "\nCOMMON FAILURE MODES\n"
            report += "-" * 22 + "\n"
            
            quality_issues = len(failure_modes["transcription_quality_issues"])
            complex_cases = len(failure_modes["complex_cases"])
            
            report += f"Transcription quality issues: {quality_issues}\n"
            report += f"Complex cases: {complex_cases}\n"
        
        return report
