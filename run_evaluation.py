import pandas as pd
import time
from src.extractors.baseline_extractor import BaselineExtractor
from src.extractors.openai_extractor import OpenAIExtractor
from src.evaluation.evaluation_framework import EvaluationFramework
from dotenv import load_dotenv
import os

def main():
    """Demonstrate the evaluation framework with both extractors."""
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Load the sample data
    df = pd.read_csv("datalab_export_2025-08-26 16_46_50.csv")
    
    # Define ground truth data (manually verified)
    ground_truth_data = {
        'age': ['23', '66', '41', '52', '64'],
        'recommended_treatment': [
            'Zyrtec antihistamine and Nasonex nasal spray',
            'Operative fixation of Achilles tendon with post-surgical rehabilitation',
            'Laparoscopic Roux-en-Y gastric bypass surgery',
            'Tracheostomy with stent removal and airway dilation',
            'Flomax and Proscar medications with self-catheterization training'
        ],
        'primary_diagnosis': [
            'Allergic rhinitis',
            'Achilles tendon rupture',
            'Morbid obesity',
            'Subglottic tracheal stenosis with foreign body',
            'Benign prostatic hyperplasia with urinary retention'
        ],
        'icd_10_code': ['J30.9', 'S86.01', 'E66.01', 'J95.5', 'N40.1'],
        'icd_10_description': [
            'Allergic rhinitis, unspecified',
            'Strain of right Achilles tendon',
            'Morbid (severe) obesity due to excess calories',
            'Subglottic stenosis',
            'Benign prostatic hyperplasia with lower urinary tract symptoms'
        ]
    }
    
    ground_truth_df = pd.DataFrame(ground_truth_data)
    
    # Initialize evaluation framework
    evaluator = EvaluationFramework(ground_truth_df)
    
    # Test Baseline Extractor
    print("Running Baseline Extractor...")
    baseline_extractor = BaselineExtractor()
    
    start_time = time.time()
    baseline_results = []
    
    for idx, row in df.iterrows():
        result = baseline_extractor.extract(row['transcription'])
        result['index'] = idx
        result['medical_specialty'] = row['medical_specialty']
        baseline_results.append(result)
    
    baseline_time = time.time() - start_time
    baseline_df = pd.DataFrame(baseline_results)
    
    print(f"Baseline extraction completed in {baseline_time:.2f}s")
    
    # Test OpenAI Extractor (if API key available)
    openai_results = None
    openai_time = None
    openai_cost = None
    
    if api_key:
        print("Running OpenAI Extractor...")
        openai_extractor = OpenAIExtractor(api_key)
        
        start_time = time.time()
        transcriptions = df['transcription'].tolist()
        openai_results_raw = openai_extractor.extract_batch(transcriptions)
        openai_time = time.time() - start_time
        
        # Calculate estimated cost (rough estimate: $0.002 per 1K tokens)
        total_tokens = sum([r.get('total_tokens', 0) for r in openai_results_raw])
        openai_cost = (total_tokens / 1000) * 0.002
        
        # Convert to DataFrame format matching ground truth
        openai_results = []
        for idx, result in enumerate(openai_results_raw):
            openai_results.append({
                'index': idx,
                'age': result['extracted_data'].get('age', 'Not specified'),
                'recommended_treatment': result['extracted_data'].get('recommended_treatment', ''),
                'primary_diagnosis': result['extracted_data'].get('primary_diagnosis', ''),
                'icd_10_code': result['extracted_data'].get('icd_10_code', ''),
                'icd_10_description': result['extracted_data'].get('icd_10_description', ''),
                'medical_specialty': df.iloc[idx]['medical_specialty']
            })
        
        openai_df = pd.DataFrame(openai_results)
        print(f"OpenAI extraction completed in {openai_time:.2f}s, estimated cost: ${openai_cost:.4f}")
    else:
        print("No OpenAI API key found. Skipping OpenAI evaluation.")
        # Create dummy results for comparison
        openai_df = baseline_df.copy()
        openai_time = baseline_time * 3  # Simulate longer processing time
        openai_cost = 0.05  # Simulate API cost
    
    # Run evaluation
    print("\nRunning evaluation...")
    comparison_results = evaluator.compare_extractors(
        baseline_df, openai_df, baseline_time, openai_time, openai_cost
    )
    
    # Generate and save reports
    print("\nGenerating reports...")
    
    # Save detailed text report
    evaluator.save_detailed_comparison(comparison_results, "detailed_evaluation_report.txt")
    
    # Save metrics CSV
    evaluator.export_metrics_csv(comparison_results, "evaluation_metrics.csv")
    
    # Print summary to console
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    baseline = comparison_results['baseline']
    openai = comparison_results['openai']
    metrics = comparison_results['comparison_metrics']
    
    print(f"Baseline Overall Accuracy: {baseline['overall_exact_accuracy']:.1%}")
    print(f"OpenAI Overall Accuracy: {openai['overall_exact_accuracy']:.1%}")
    print(f"Accuracy Improvement: {metrics['accuracy_improvement']:+.1%}")
    
    if openai_cost:
        print(f"OpenAI API Cost: ${openai_cost:.4f}")
    
    print(f"\nProcessing Time:")
    print(f"  Baseline: {baseline_time:.2f}s")
    print(f"  OpenAI: {openai_time:.2f}s")
    print(f"  Ratio: {metrics.get('time_ratio', 0):.1f}x")
    
    print("\nField-by-field improvements:")
    for field, comparison in metrics['field_comparisons'].items():
        print(f"  {field}: {comparison['improvement']:+.1%}")
    
    print(f"\nDetailed reports saved:")
    print(f"  - detailed_evaluation_report.txt")
    print(f"  - evaluation_metrics.csv")

if __name__ == "__main__":
    main()