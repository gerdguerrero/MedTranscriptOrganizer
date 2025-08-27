import re
from typing import Dict, Tuple, Any

class BaselineExtractor:
    """
    Baseline medical information extractor using regex patterns.
    Extracts age, recommended treatment, primary diagnosis, ICD-10 code, and description from transcription text.
    Includes confidence scoring based on pattern matches.
    """
    def __init__(self):
        # Define regex patterns for age
        self.age_patterns = [
            r'(\d+)[-\s]year[-\s]old',
            r'(\d+)\s+years?\s+of\s+age',
            r'age\s+(\d+)',
            r'(\d+)[-\s]yo'
        ]
        
        # Define regex patterns for treatment recommendations
        self.treatment_patterns = [
            r'plan\s*:\s*([^.]+)',
            r'treatment\s*:\s*([^.]+)',
            r'prescribed\s+([^.]+)',
            r'recommend\w*\s+([^.]+)',
            r'therapy\s*:\s*([^.]+)',
            r'medication\s*:\s*([^.]+)'
        ]
        
        # Define regex patterns for primary diagnosis
        self.diagnosis_patterns = [
            r'diagnosis\s*:\s*([^.]+)',
            r'impression\s*:\s*([^.]+)',
            r'assessment\s*:\s*([^.]+)',
            r'condition\s*:\s*([^.]+)',
            r'findings\s*:\s*([^.]+)'
        ]
        
        # Simple ICD-10 code mapping based on common medical terms
        self.icd10_mapping = {
            # Respiratory conditions
            'allergic rhinitis': ('J30.9', 'Allergic rhinitis, unspecified'),
            'asthma': ('J45.9', 'Asthma, unspecified'),
            'pneumonia': ('J18.9', 'Pneumonia, unspecified organism'),
            'bronchitis': ('J40', 'Bronchitis, not specified as acute or chronic'),
            
            # Cardiovascular conditions
            'hypertension': ('I10', 'Essential hypertension'),
            'heart failure': ('I50.9', 'Heart failure, unspecified'),
            'atrial fibrillation': ('I48.91', 'Unspecified atrial fibrillation'),
            
            # Gastrointestinal conditions
            'gastritis': ('K29.70', 'Gastritis, unspecified, without bleeding'),
            'gerd': ('K21.9', 'Gastro-esophageal reflux disease without esophagitis'),
            'gastroesophageal reflux': ('K21.9', 'Gastro-esophageal reflux disease without esophagitis'),
            
            # Musculoskeletal conditions
            'arthritis': ('M19.90', 'Unspecified osteoarthritis, unspecified site'),
            'back pain': ('M54.9', 'Dorsalgia, unspecified'),
            'achilles tendon': ('S86.01', 'Strain of right Achilles tendon'),
            
            # Endocrine conditions
            'diabetes': ('E11.9', 'Type 2 diabetes mellitus without complications'),
            'obesity': ('E66.9', 'Obesity, unspecified'),
            'morbid obesity': ('E66.01', 'Morbid (severe) obesity due to excess calories'),
            
            # Urological conditions
            'prostate': ('N40.1', 'Benign prostatic hyperplasia with lower urinary tract symptoms'),
            'benign prostatic hyperplasia': ('N40.1', 'Benign prostatic hyperplasia with lower urinary tract symptoms'),
            
            # Other conditions
            'stenosis': ('J95.5', 'Subglottic stenosis'),
            'tracheal stenosis': ('J95.5', 'Subglottic stenosis')
        }

    def extract_age(self, text: str) -> Tuple[str, float]:
        """Extract age with confidence score."""
        for i, pattern in enumerate(self.age_patterns):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Higher confidence for more specific patterns
                confidence = 0.9 - (i * 0.1)
                return match.group(1), confidence
        return "Not specified", 0.0

    def extract_treatment(self, text: str) -> Tuple[str, float]:
        """Extract treatment recommendations with confidence score."""
        best_match = ""
        best_confidence = 0.0
        
        for i, pattern in enumerate(self.treatment_patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Use the longest match and adjust confidence based on pattern specificity
                current_match = max(matches, key=len).strip()
                confidence = 0.8 - (i * 0.1)
                
                if len(current_match) > len(best_match):
                    best_match = current_match
                    best_confidence = confidence
        
        return best_match if best_match else "Not specified", best_confidence

    def extract_diagnosis(self, text: str) -> Tuple[str, float]:
        """Extract primary diagnosis with confidence score."""
        best_match = ""
        best_confidence = 0.0
        
        for i, pattern in enumerate(self.diagnosis_patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Use the longest match and adjust confidence based on pattern specificity
                current_match = max(matches, key=len).strip()
                confidence = 0.8 - (i * 0.1)
                
                if len(current_match) > len(best_match):
                    best_match = current_match
                    best_confidence = confidence
        
        return best_match if best_match else "Not specified", best_confidence

    def extract_icd10_code(self, diagnosis_text: str) -> Tuple[str, str, float]:
        """Extract ICD-10 code based on diagnosis text with confidence score."""
        diagnosis_lower = diagnosis_text.lower()
        
        for condition, (code, description) in self.icd10_mapping.items():
            if condition in diagnosis_lower:
                # Higher confidence for more specific matches
                confidence = 0.7 if len(condition.split()) > 1 else 0.5
                return code, description, confidence
        
        return "Not specified", "No matching ICD-10 code found", 0.0

    def calculate_overall_confidence(self, confidences: Dict[str, float]) -> float:
        """Calculate overall extraction confidence."""
        valid_confidences = [conf for conf in confidences.values() if conf > 0]
        if not valid_confidences:
            return 0.0
        return sum(valid_confidences) / len(valid_confidences)

    def extract(self, transcription: str) -> Dict[str, Any]:
        """Extract all medical information with confidence scores."""
        # Extract individual components
        age, age_conf = self.extract_age(transcription)
        treatment, treatment_conf = self.extract_treatment(transcription)
        diagnosis, diagnosis_conf = self.extract_diagnosis(transcription)
        icd10_code, icd10_desc, icd10_conf = self.extract_icd10_code(diagnosis)
        
        # Calculate overall confidence
        confidences = {
            'age': age_conf,
            'treatment': treatment_conf,
            'diagnosis': diagnosis_conf,
            'icd10': icd10_conf
        }
        overall_confidence = self.calculate_overall_confidence(confidences)
        
        return {
            "age": age,
            "recommended_treatment": treatment,
            "primary_diagnosis": diagnosis,
            "icd_10_code": icd10_code,
            "icd_10_description": icd10_desc,
            "confidence_scores": confidences,
            "overall_confidence": overall_confidence
        }
