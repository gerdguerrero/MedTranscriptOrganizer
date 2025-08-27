"""
Medical Transcription Analysis Comparison Tool

This script compares the baseline regex-based extractor with the OpenAI API extractor
and provides comprehensive evaluation metrics and error analysis.
"""

import pandas as pd
import os
from dotenv import load_dotenv
import sys

# Add src to path to import our modules
sys.path.append('src')

from extractors.baseline_extractor import BaselineExtractor
from extractors.openai_extractor import OpenAIExtractor
from evaluation.metrics import EvaluationMetrics
from evaluation.error_analysis import ErrorAnalysis

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the medical transcription data."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} transcriptions from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return None

def run_baseline_extraction(df: pd.DataFrame) -> pd.DataFrame:
    """Run baseline extraction on all transcriptions."""
    print("Running baseline extraction...")
    
    extractor = BaselineExtractor()
    results = []
    
    for idx, row in df.iterrows():
        transcription = row['transcription']
        specialty = row['medical_specialty']
        
        # Extract information
        extracted = extractor.extract(transcription)
        
        # Create result record
        result = {
            'index': idx,
            'medical_specialty': specialty,
            'age': extracted['age'],
            'recommended_treatment': extracted['recommended_treatment'],
            'primary_diagnosis': extracted['primary_diagnosis'],
            'icd_10_code': extracted['icd_10_code'],
            'icd_10_description': extracted['icd_10_description'],
            'overall_confidence': extracted['overall_confidence']
        }
        results.append(result)
    
    print(f"Baseline extraction completed for {len(results)} transcriptions")
    return pd.DataFrame(results)

def run_openai_extraction(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """Run OpenAI extraction on all transcriptions."""
    print("Running OpenAI extraction...")
    
    try:
        extractor = OpenAIExtractor(api_key)
        results = []
        
        for idx, row in df.iterrows():
            transcription = row['transcription']
            specialty = row['medical_specialty']
            
            print(f"Processing transcription {idx + 1}/{len(df)}")
            
            # Extract information
            extracted = extractor.extract(transcription)
            
            # Create result record
            result = {
                'index': idx,
                'medical_specialty': specialty,
                'age': extracted['age'],
                'recommended_treatment': extracted['recommended_treatment'],
                'primary_diagnosis': extracted['primary_diagnosis'],
                'icd_10_code': extracted['icd_10_code'],
                'icd_10_description': extracted['icd_10_description'],
                'overall_confidence': extracted['overall_confidence']
            }
            results.append(result)
        
        print(f"OpenAI extraction completed for {len(results)} transcriptions")
        return pd.DataFrame(results)
        
    except Exception as e:
        print(f"Error during OpenAI extraction: {e}")
        return None

def create_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ground truth data for evaluation.
    In a real scenario, this would be manually annotated data.
    For demonstration, we'll use the manual extraction from the original script.
    """
    print("Creating ground truth data...")
    
    ground_truth = []
    
    for idx, row in df.iterrows():
        # Manual ground truth based on analysis of each case
        if idx == 0:  # Allergy case
            gt = {
                'index': idx,
                'age': '23',
                'recommended_treatment': 'Zyrtec antihistamine and Nasonex nasal spray',
                'primary_diagnosis': 'Allergic rhinitis',
                'icd_10_code': 'J30.9'
            }
        elif idx == 1:  # Orthopedic case
            gt = {
                'index': idx,
                'age': '66',
                'recommended_treatment': 'Operative fixation of Achilles tendon with post-surgical rehabilitation',
                'primary_diagnosis': 'Achilles tendon rupture',
                'icd_10_code': 'S86.01'
            }
        elif idx == 2:  # Bariatric case
            gt = {
                'index': idx,
                'age': 'Not specified',
                'recommended_treatment': 'Laparoscopic Roux-en-Y gastric bypass surgery',
                'primary_diagnosis': 'Morbid obesity',
                'icd_10_code': 'E66.01'
            }
        elif idx == 3:  # Cardiovascular case
            gt = {
                'index': idx,
                'age': 'Not specified',
                'recommended_treatment': 'Tracheostomy with stent removal and airway dilation',
                'primary_diagnosis': 'Subglottic tracheal stenosis with foreign body',
                'icd_10_code': 'J95.5'
            }
        elif idx == 4:  # Urology case
            gt = {
                'index': idx,
                'age': 'Not specified',
                'recommended_treatment': 'Flomax and Proscar medications with self-catheterization training',
                'primary_diagnosis': 'Benign prostatic hyperplasia with urinary retention',
                'icd_10_code': 'N40.1'
            }
        else:
            # Default for any additional cases
            gt = {
                'index': idx,
                'age': 'Not specified',
                'recommended_treatment': 'Not specified',
                'primary_diagnosis': 'Not specified',
                'icd_10_code': 'Not specified'
            }
        
        ground_truth.append(gt)
    
    return pd.DataFrame(ground_truth)

def main():
    """Main comparison workflow."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Load data
    csv_path = "datalab_export_2025-08-26 16_46_50.csv"
    df = load_data(csv_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create ground truth
    ground_truth = create_ground_truth(df)
    
    # Run baseline extraction
    baseline_results = run_baseline_extraction(df)
    
    # Run OpenAI extraction (if API key is available)
    openai_results = None
    if api_key:
        openai_results = run_openai_extraction(df, api_key)
    else:
        print("No OpenAI API key found. Skipping OpenAI extraction.")
    
    # Save results
    baseline_results.to_csv('baseline_extraction_results.csv', index=False)
    print("Baseline results saved to 'baseline_extraction_results.csv'")
    
    if openai_results is not None:
        openai_results.to_csv('openai_extraction_results.csv', index=False)
        print("OpenAI results saved to 'openai_extraction_results.csv'")
    
    ground_truth.to_csv('ground_truth_annotations.csv', index=False)
    print("Ground truth saved to 'ground_truth_annotations.csv'")
    
    # Evaluation and comparison
    if openai_results is not None:
        print("\n" + "="*60)
        print("EVALUATION AND COMPARISON")
        print("="*60)
        
        # Calculate metrics
        evaluator = EvaluationMetrics()
        comparison = evaluator.compare_extractors(baseline_results, openai_results, ground_truth)
        
        # Generate and display report
        report = evaluator.generate_report(comparison)
        print(report)
        
        # Error analysis
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        
        error_analyzer = ErrorAnalysis()
        
        # Analyze baseline errors
        baseline_error_analysis = error_analyzer.analyze_errors(
            baseline_results, ground_truth, df['transcription'].tolist()
        )
        
        # Analyze OpenAI errors
        openai_error_analysis = error_analyzer.analyze_errors(
            openai_results, ground_truth, df['transcription'].tolist()
        )
        
        # Generate error reports
        print("\nBASELINE EXTRACTOR ERRORS:")
        print("-" * 30)
        baseline_error_report = error_analyzer.generate_error_report(baseline_error_analysis)
        print(baseline_error_report)
        
        print("\nOPENAI EXTRACTOR ERRORS:")
        print("-" * 25)
        openai_error_report = error_analyzer.generate_error_report(openai_error_analysis)
        print(openai_error_report)
        
        # Save detailed results
        with open('evaluation_report.txt', 'w') as f:
            f.write(report + "\n\n")
            f.write("BASELINE EXTRACTOR ERRORS:\n")
            f.write("-" * 30 + "\n")
            f.write(baseline_error_report + "\n\n")
            f.write("OPENAI EXTRACTOR ERRORS:\n")
            f.write("-" * 25 + "\n")
            f.write(openai_error_report)
        
        print("\nDetailed evaluation report saved to 'evaluation_report.txt'")
    
    else:
        # Just show baseline results
        print("\n" + "="*60)
        print("BASELINE EXTRACTION RESULTS")
        print("="*60)
        print(baseline_results)
        
        print(f"\nBaseline extraction completed.")
        print(f"Results saved to 'baseline_extraction_results.csv'")

if __name__ == "__main__":
    main()
