"""
Comprehensive Error Analysis for Medical Transcription Extraction
"""

import pandas as pd
import os
from dotenv import load_dotenv
from src.evaluation.error_analyzer import ErrorAnalyzer
from src.extractors.baseline_extractor import BaselineExtractor
from src.extractors.openai_extractor import OpenAIExtractor

def create_ground_truth_from_csv():
    """Create ground truth DataFrame from your structured data."""
    # Based on your structured_medical_data.csv, create corrected ground truth
    ground_truth_data = {
        'age': ['23', '41', '30', '50', '66'],  # Corrected based on your data
        'recommended_treatment': [
            'Zyrtec antihistamine with Nasonex nasal spray',
            'Operative fixation of right Achilles tendon with rehabilitation',
            'Laparoscopic Roux-en-Y gastric bypass surgery',
            'Tracheostomy with foreign body removal and airway dilation',
            'Flomax and Proscar with self-catheterization training'
        ],
        'primary_diagnosis': [
            'Allergic rhinitis',
            'Right Achilles tendon rupture',
            'Morbid obesity',
            'Subglottic tracheal stenosis with foreign body',
            'Benign prostatic hyperplasia'
        ],
        'icd_10_code': ['J30.1', 'S86.011A', 'E66.01', 'J39.8', 'N40.1'],
        'icd_10_description': [
            'Allergic rhinitis due to pollen',
            'Complete traumatic rupture of right Achilles tendon, initial encounter',
            'Morbid (severe) obesity due to excess calories',
            'Other specified diseases of upper respiratory tract',
            'Benign prostatic hyperplasia'
        ]
    }
    
    return pd.DataFrame(ground_truth_data)

def analyze_existing_results():
    """Analyze the existing structured_medical_data.csv results."""
    print("=" * 60)
    print("ANALYZING EXISTING EXTRACTION RESULTS")
    print("=" * 60)
    
    # Load your existing results
    if not os.path.exists('structured_medical_data.csv'):
        print("Error: structured_medical_data.csv not found!")
        print("Please run medical_analysis.py first to generate results.")
        return
    
    results_df = pd.read_csv('structured_medical_data.csv')
    ground_truth_df = create_ground_truth_from_csv()
    
    print(f"Loaded {len(results_df)} results and {len(ground_truth_df)} ground truth records")
    
    # Initialize error analyzer
    analyzer = ErrorAnalyzer(results_df, ground_truth_df)
    
    # Run comprehensive analysis
    print("\nRunning comprehensive error analysis...")
    error_analysis = analyzer.run_complete_analysis()
    
    # Generate insights
    print("\nGenerating actionable insights...")
    insights = analyzer.generate_actionable_insights()
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("=" * 60)
    for insight in insights:
        print(f"• {insight}")
    
    # Generate detailed report
    analyzer.generate_detailed_report("existing_results_error_analysis.txt")
    
    # Create visualizations
    print("\nCreating error visualizations...")
    viz_result = analyzer.create_error_visualizations()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS:")
    print("=" * 60)
    summary = error_analysis['summary_statistics']
    print(f"Overall Accuracy: {summary['overall_accuracy']:.1%}")
    print(f"Best Field: {summary['best_performing_field']}")
    print(f"Worst Field: {summary['most_problematic_field']}")
    
    print("\nField Performance:")
    for field, perf in summary['field_performance'].items():
        print(f"  {field}: {perf['accuracy']:.1%} accuracy, {perf['total_errors']} errors")
    
    return analyzer

def compare_baseline_vs_openai():
    """Compare baseline extractor vs OpenAI extractor if both are available."""
    print("\n" + "=" * 60)
    print("COMPARING BASELINE VS OPENAI EXTRACTORS")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Load original data
    if not os.path.exists("datalab_export_2025-08-26 16_46_50.csv"):
        print("Original data file not found. Skipping extractor comparison.")
        return
    
    df = pd.read_csv("datalab_export_2025-08-26 16_46_50.csv")
    ground_truth_df = create_ground_truth_from_csv()
    
    # Test baseline extractor
    print("Testing baseline extractor...")
    baseline_extractor = BaselineExtractor()
    baseline_results = []
    
    for idx, row in df.iterrows():
        result = baseline_extractor.extract(row['transcription'])
        baseline_results.append({
            'age': result.get('age', 'Not specified'),
            'recommended_treatment': result.get('recommended_treatment', ''),
            'primary_diagnosis': result.get('primary_diagnosis', ''),
            'icd_10_code': result.get('icd_10_code', ''),
            'icd_10_description': result.get('icd_10_description', '')
        })
    
    baseline_df = pd.DataFrame(baseline_results)
    
    # Test OpenAI extractor (if available)
    if api_key:
        print("Testing OpenAI extractor...")
        openai_extractor = OpenAIExtractor(api_key)
        transcriptions = df['transcription'].tolist()
        openai_results_raw = openai_extractor.extract_batch(transcriptions)
        
        openai_results = []
        for result in openai_results_raw:
            data = result['extracted_data']
            openai_results.append({
                'age': data.get('age', 'Not specified'),
                'recommended_treatment': data.get('recommended_treatment', ''),
                'primary_diagnosis': data.get('primary_diagnosis', ''),
                'icd_10_code': data.get('icd_10_code', ''),
                'icd_10_description': data.get('icd_10_description', '')
            })
        
        openai_df = pd.DataFrame(openai_results)
        
        # Compare extractors
        analyzer = ErrorAnalyzer(baseline_df, ground_truth_df)
        comparison = analyzer.compare_extractors(baseline_df, openai_df)
        
        print(f"\nComparison Results:")
        print(f"Cases where baseline is better: {len(comparison['baseline_better'])}")
        print(f"Cases where OpenAI is better: {len(comparison['openai_better'])}")
        print(f"Cases where both are wrong: {len(comparison['both_wrong'])}")
        print(f"Cases where both are correct: {len(comparison['both_correct'])}")
        
        # Show specific examples
        if comparison['baseline_better']:
            print(f"\nExample where baseline outperformed OpenAI:")
            example = comparison['baseline_better'][0]
            print(f"  Field: {example['field']}")
            print(f"  Actual: {example['actual']}")
            print(f"  Baseline: {example['baseline']} (similarity: {example['baseline_similarity']:.2f})")
            print(f"  OpenAI: {example['openai']} (similarity: {example['openai_similarity']:.2f})")
        
        if comparison['openai_better']:
            print(f"\nExample where OpenAI outperformed baseline:")
            example = comparison['openai_better'][0]
            print(f"  Field: {example['field']}")
            print(f"  Actual: {example['actual']}")
            print(f"  Baseline: {example['baseline']} (similarity: {example['baseline_similarity']:.2f})")
            print(f"  OpenAI: {example['openai']} (similarity: {example['openai_similarity']:.2f})")
        
    else:
        print("No OpenAI API key found. Analyzing baseline extractor only...")
        analyzer = ErrorAnalyzer(baseline_df, ground_truth_df)
        analyzer.run_complete_analysis()
        analyzer.generate_detailed_report("baseline_error_analysis.txt")

def main():
    """Main function to run error analysis."""
    print("Medical Transcription Error Analysis Tool")
    print("=" * 60)
    
    print("\nSelect analysis option:")
    print("1. Analyze existing structured_medical_data.csv")
    print("2. Compare baseline vs OpenAI extractors")
    print("3. Run both analyses")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        analyze_existing_results()
    elif choice == "2":
        compare_baseline_vs_openai()
    elif choice == "3":
        analyze_existing_results()
        compare_baseline_vs_openai()
    else:
        print("Invalid choice. Running analysis of existing results...")
        analyze_existing_results()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("Check the following files for detailed results:")
    print("• existing_results_error_analysis.txt")
    print("• error_analysis_visualizations.png")
    print("• baseline_error_analysis.txt (if generated)")
    print("=" * 60)

if __name__ == "__main__":
    main()