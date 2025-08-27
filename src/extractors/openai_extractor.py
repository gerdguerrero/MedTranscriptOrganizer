import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
import tiktoken
from datetime import datetime
import random

class OpenAIExtractor:
    """
    Enhanced OpenAI API-based medical information extractor with:
    - Structured prompts for consistency
    - Confidence scoring from OpenAI
    - Retry logic and error handling
    - Processing metadata (time, tokens)
    - Efficient batch processing
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", max_retries: int = 3, 
                 base_delay: float = 1.0, max_delay: float = 60.0):
        """
        Initialize the enhanced OpenAI extractor.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for extraction
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Structured prompts for consistency
        self.system_prompt = """You are a medical coding specialist with expertise in ICD-10 codes and medical terminology. 
        
Your task is to analyze medical transcriptions and extract key information with high accuracy. 
Always provide confidence scores for your extractions on a scale of 0.0 to 1.0, where:
- 1.0 = Completely certain, information explicitly stated
- 0.8-0.9 = High confidence, clear medical terminology
- 0.6-0.7 = Moderate confidence, reasonable inference
- 0.4-0.5 = Low confidence, limited information
- 0.0-0.3 = Very uncertain, minimal or unclear information

Respond ONLY in valid JSON format without any markdown formatting."""

        self.extraction_prompt_template = """Analyze the medical transcription below and extract the following information:

1. **Patient Age**: Extract the patient's age if mentioned (or "Not specified")
2. **Recommended Treatment**: Main treatment plan, medications, or procedures recommended
3. **Primary Diagnosis**: The main medical condition or diagnosis
4. **ICD-10 Code**: Most appropriate ICD-10 code for the primary diagnosis
5. **Confidence Scores**: Rate your confidence in each extraction (0.0-1.0)

Medical Transcription:
{transcription}

Respond in this exact JSON format:
{{
    "age": "patient age or 'Not specified'",
    "recommended_treatment": "detailed treatment plan",
    "primary_diagnosis": "main medical diagnosis",
    "icd_10_code": "appropriate ICD-10 code",
    "icd_10_description": "description of the ICD-10 code",
    "confidence_scores": {{
        "age": 0.0,
        "treatment": 0.0,
        "diagnosis": 0.0,
        "icd10": 0.0
    }},
    "overall_confidence": 0.0,
    "extraction_notes": "any relevant notes about the extraction"
}}"""

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
        return min(delay, self.max_delay)
    
    def _make_api_call(self, messages: List[Dict[str, str]], max_tokens: int = 800) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Make API call with retry logic and metadata collection.
        
        Returns:
            Tuple of (result, metadata)
        """
        start_time = time.time()
        
        # Count input tokens
        input_text = " ".join([msg["content"] for msg in messages])
        input_tokens = self._count_tokens(input_text)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    top_p=0.9
                )
                
                # Extract response content
                content = response.choices[0].message.content
                output_tokens = self._count_tokens(content) if content else 0
                
                # Process time
                processing_time = time.time() - start_time
                
                # Parse JSON response
                result = self._parse_json_response(content)
                
                # Create metadata
                metadata = {
                    "processing_time_seconds": processing_time,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "model_used": self.model,
                    "attempt_number": attempt + 1,
                    "api_call_successful": True,
                    "timestamp": datetime.now().isoformat()
                }
                
                return result, metadata
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON parsing error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return self._get_error_response("JSON parsing failed after retries"), {
                        "processing_time_seconds": time.time() - start_time,
                        "api_call_successful": False,
                        "error_type": "json_decode_error",
                        "final_attempt": attempt + 1
                    }
                    
            except Exception as e:
                self.logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    return self._get_error_response(f"API error: {str(e)}"), {
                        "processing_time_seconds": time.time() - start_time,
                        "api_call_successful": False,
                        "error_type": "api_error",
                        "final_attempt": attempt + 1
                    }
        
        return self._get_error_response("Max retries exceeded"), {
            "processing_time_seconds": time.time() - start_time,
            "api_call_successful": False,
            "error_type": "max_retries_exceeded"
        }
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response with cleaning."""
        if not content:
            raise json.JSONDecodeError("Empty response", "", 0)
        
        # Clean the response
        content = content.strip()
        
        # Remove markdown formatting if present
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        content = content.strip()
        
        # Parse JSON
        result = json.loads(content)
        
        # Validate required fields
        required_fields = ["age", "recommended_treatment", "primary_diagnosis", 
                          "icd_10_code", "icd_10_description", "confidence_scores", "overall_confidence"]
        
        for field in required_fields:
            if field not in result:
                if field == "confidence_scores":
                    result[field] = {"age": 0.5, "treatment": 0.5, "diagnosis": 0.5, "icd10": 0.5}
                elif field == "overall_confidence":
                    result[field] = 0.5
                else:
                    result[field] = "Not specified"
        
        # Ensure confidence scores are valid
        if "confidence_scores" in result:
            for key, value in result["confidence_scores"].items():
                if not isinstance(value, (int, float)) or value < 0 or value > 1:
                    result["confidence_scores"][key] = 0.5
        
        # Calculate overall confidence if not provided or invalid
        if "overall_confidence" not in result or not isinstance(result["overall_confidence"], (int, float)):
            conf_values = list(result["confidence_scores"].values())
            result["overall_confidence"] = sum(conf_values) / len(conf_values) if conf_values else 0.5
        
        return result
    
    def _get_error_response(self, error_message: str) -> Dict[str, Any]:
        """Return a standardized error response."""
        return {
            "age": "Error extracting age",
            "recommended_treatment": "Error extracting treatment",
            "primary_diagnosis": "Error extracting diagnosis",
            "icd_10_code": "N/A",
            "icd_10_description": f"Error occurred: {error_message}",
            "confidence_scores": {"age": 0.0, "treatment": 0.0, "diagnosis": 0.0, "icd10": 0.0},
            "overall_confidence": 0.0,
            "extraction_notes": f"Extraction failed: {error_message}"
        }
    
    def extract(self, transcription: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract medical information from a single transcription.
        
        Args:
            transcription: Medical transcription text
            
        Returns:
            Tuple of (extraction_result, metadata)
        """
        # Prepare messages
        user_prompt = self.extraction_prompt_template.format(transcription=transcription)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Make API call
        result, metadata = self._make_api_call(messages)
        
        # Add transcription metadata
        metadata["transcription_length"] = len(transcription)
        metadata["transcription_word_count"] = len(transcription.split())
        
        return result, metadata
    
    def extract_batch(self, transcriptions: List[str], batch_size: int = 5, 
                     delay_between_batches: float = 2.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract information from multiple transcriptions efficiently.
        
        Args:
            transcriptions: List of transcription texts
            batch_size: Number of transcriptions to process in parallel batches
            delay_between_batches: Delay between batches in seconds
            
        Returns:
            Tuple of (list_of_results, batch_metadata)
        """
        self.logger.info(f"Starting batch extraction of {len(transcriptions)} transcriptions")
        
        start_time = time.time()
        results = []
        individual_metadata = []
        
        # Process in batches
        for i in range(0, len(transcriptions), batch_size):
            batch = transcriptions[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(transcriptions) + batch_size - 1) // batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} transcriptions)")
            
            # Process each transcription in the batch
            for j, transcription in enumerate(batch):
                transcription_num = i + j + 1
                self.logger.info(f"Processing transcription {transcription_num}/{len(transcriptions)}")
                
                result, metadata = self.extract(transcription)
                results.append(result)
                individual_metadata.append(metadata)
                
                # Small delay between individual extractions within a batch
                if j < len(batch) - 1:
                    time.sleep(0.5)
            
            # Delay between batches (except for the last batch)
            if i + batch_size < len(transcriptions):
                self.logger.info(f"Waiting {delay_between_batches} seconds before next batch...")
                time.sleep(delay_between_batches)
        
        # Calculate batch metadata
        total_time = time.time() - start_time
        successful_extractions = sum(1 for meta in individual_metadata if meta.get("api_call_successful", False))
        
        # Aggregate token usage
        total_input_tokens = sum(meta.get("input_tokens", 0) for meta in individual_metadata)
        total_output_tokens = sum(meta.get("output_tokens", 0) for meta in individual_metadata)
        
        # Aggregate confidence scores
        valid_confidences = [result.get("overall_confidence", 0) for result in results 
                           if isinstance(result.get("overall_confidence"), (int, float))]
        avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
        
        batch_metadata = {
            "total_transcriptions": len(transcriptions),
            "successful_extractions": successful_extractions,
            "failed_extractions": len(transcriptions) - successful_extractions,
            "success_rate": successful_extractions / len(transcriptions),
            "total_processing_time_seconds": total_time,
            "average_time_per_transcription": total_time / len(transcriptions),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "average_confidence": avg_confidence,
            "batch_size_used": batch_size,
            "delay_between_batches": delay_between_batches,
            "model_used": self.model,
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Batch extraction completed. Success rate: {batch_metadata['success_rate']:.1%}")
        
        return results, batch_metadata
    
    def get_extraction_summary(self, results: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        """Generate a summary report of the extraction process."""
        
        report = "OpenAI Extraction Summary Report\n"
        report += "=" * 40 + "\n\n"
        
        # Basic statistics
        report += f"Total transcriptions processed: {metadata['total_transcriptions']}\n"
        report += f"Successful extractions: {metadata['successful_extractions']}\n"
        report += f"Failed extractions: {metadata['failed_extractions']}\n"
        report += f"Success rate: {metadata['success_rate']:.1%}\n\n"
        
        # Performance metrics
        report += f"Total processing time: {metadata['total_processing_time_seconds']:.2f} seconds\n"
        report += f"Average time per transcription: {metadata['average_time_per_transcription']:.2f} seconds\n\n"
        
        # Token usage
        report += f"Total tokens used: {metadata['total_tokens']:,}\n"
        report += f"Input tokens: {metadata['total_input_tokens']:,}\n"
        report += f"Output tokens: {metadata['total_output_tokens']:,}\n\n"
        
        # Quality metrics
        report += f"Average confidence score: {metadata['average_confidence']:.3f}\n"
        
        # Confidence distribution
        confidences = [r.get("overall_confidence", 0) for r in results if isinstance(r.get("overall_confidence"), (int, float))]
        if confidences:
            high_conf = sum(1 for c in confidences if c > 0.8)
            med_conf = sum(1 for c in confidences if 0.5 <= c <= 0.8)
            low_conf = sum(1 for c in confidences if c < 0.5)
            
            report += f"High confidence (>0.8): {high_conf} ({high_conf/len(confidences):.1%})\n"
            report += f"Medium confidence (0.5-0.8): {med_conf} ({med_conf/len(confidences):.1%})\n"
            report += f"Low confidence (<0.5): {low_conf} ({low_conf/len(confidences):.1%})\n"
        
        return report
