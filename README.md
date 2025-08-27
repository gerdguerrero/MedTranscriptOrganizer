# Medical Transcription Analysis Project

A comprehensive medical transcription analysis tool that compares baseline regex-based extraction with OpenAI API-powered extraction, including evaluation metrics and error analysis.

## Project Structure

```
MedTranscriptOrganizer/
├── src/
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── baseline_extractor.py    # Regex-based extractor
│   │   └── openai_extractor.py      # OpenAI API extractor
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py               # Evaluation metrics
│       └── error_analysis.py        # Error analysis tools
├── compare_extractors.py            # Main comparison script
├── medical_analysis.py              # Original analysis script
├── requirements.txt
├── .env                            # Environment variables (API keys)
└── README.md
```

## Features

### Extractors

1. **BaselineExtractor**: Regex-based medical information extraction
   - Age extraction from multiple pattern formats
   - Treatment recommendations using keyword patterns
   - Primary diagnosis extraction
   - Simple ICD-10 code mapping
   - Confidence scoring based on pattern matches

2. **OpenAIExtractor**: GPT-powered medical information extraction
   - Comprehensive medical information extraction
   - Advanced ICD-10 code suggestions
   - Confidence scoring from the model
   - Batch processing with rate limiting

### Evaluation Metrics

- **Exact Match Accuracy**: Perfect string matching
- **Partial Match Scoring**: Token-based overlap scoring
- **ICD-10 Specific Metrics**: 
  - Exact code matching
  - Category-level matching (first 3 characters)
  - Chapter-level matching (first character)
- **Confidence Calibration**: Expected and Maximum Calibration Error

### Error Analysis

- **Error Categorization**:
  - Missing information
  - Incorrect extraction
  - Partial extraction
  - Format errors
  - Confidence miscalibration
- **Failure Mode Identification**:
  - Transcription quality issues
  - Complex cases
  - Systematic errors
- **Detailed Error Reporting**

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in `.env`:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Basic Comparison
```bash
python compare_extractors.py
```

This will:
1. Load medical transcription data
2. Run both extractors
3. Compare results against ground truth
4. Generate evaluation reports
5. Perform detailed error analysis

### Individual Extractor Usage

```python
from src.extractors.baseline_extractor import BaselineExtractor
from src.extractors.openai_extractor import OpenAIExtractor

# Baseline extraction
baseline = BaselineExtractor()
result = baseline.extract("Patient is a 65-year-old male with diabetes...")

# OpenAI extraction
openai_extractor = OpenAIExtractor(api_key="your-key")
result = openai_extractor.extract("Patient is a 65-year-old male with diabetes...")
```

## Output Files

- `baseline_extraction_results.csv`: Results from regex-based extraction
- `openai_extraction_results.csv`: Results from OpenAI extraction
- `ground_truth_annotations.csv`: Ground truth data for evaluation
- `evaluation_report.txt`: Comprehensive comparison and error analysis

## Evaluation Metrics

The system provides multiple evaluation metrics:

- **Field-level accuracy** for age, treatment, diagnosis, and ICD-10 codes
- **Confidence calibration** analysis
- **Error type breakdown** and failure mode identification
- **Comparative analysis** between extraction methods

## Extensions

The modular design allows for easy extension:

- Add new extraction methods
- Implement additional evaluation metrics
- Extend error analysis capabilities
- Support for different medical data formats

---

## Original Next.js Project

This project was originally bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
