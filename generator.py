import os
import json
import time
from typing import Dict, Any, List
from google import genai
from google.genai import types

from config import GEMINI_MODEL_NAME, SYSTEM_INSTRUCTION


class GenerationResult:
    """Encapsulates raw API call outcomes and retry details."""
    def __init__(self, success: bool, raw_text: str, attempts: int, error: str = ""):
        self.success = success
        self.raw_text = raw_text
        self.attempts = attempts
        self.error = error


def build_context_block(evidence: List[Dict[str, Any]]) -> str:
    """Formats retrieved vector store chunks into a structured context string."""
    blocks = []
    for e in evidence:
        blocks.append(
            f"--- SOURCE CHUNK ID: {e.get('chunk_id', 'UNKNOWN')} ---\n"
            f"Document: {e.get('document_name', 'N/A')} | Page: {e.get('page', 'N/A')} | Section: {e.get('section', 'N/A')}\n"
            f"Content: {e.get('text', '')}\n"
        )
    return "\n".join(blocks)


def generate_answer_with_backoff(query: str, context_block: str, max_retries: int = 3) -> GenerationResult:
    """Executes Gemini API generation with exponential backoff retry logic."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        error_msg = "GEMINI_API_KEY environment variable is not set in environment or .env file."
        print(f"\n[Gemini API Error]: {error_msg}\n")
        return GenerationResult(False, "", 0, error_msg)

    client = genai.Client(api_key=api_key)
    user_prompt = f"Context Evidence:\n{context_block}\n\nQuestion: {query}"
    
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            return GenerationResult(True, response.text, attempt, "")
            
        except Exception as err:
            print(f"[Gemini API Attempt {attempt}/{max_retries} Failed]: {err}")
            if attempt == max_retries:
                return GenerationResult(False, "", attempt, str(err))
            
            time.sleep(delay)
            delay *= 2.0

    return GenerationResult(False, "", max_retries, "Unknown generation failure")


def generate_answer(query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Primary pipeline interface: builds context, calls model, and parses structured output."""
    context_block = build_context_block(context_chunks)
    result = generate_answer_with_backoff(query, context_block)

    if not result.success:
        return {
            "status": "ERROR",
            "answer": f"Model generation failed: {result.error}",
            "source_ids": [],
            "grounding": "UNSUPPORTED",
            "notes": f"Failed after {result.attempts} attempts."
        }

    try:
        # Parse model output JSON
        parsed_json = json.loads(result.raw_text)
        parsed_json["status"] = "SUCCESS"
        return parsed_json
        
    except json.JSONDecodeError as json_err:
        print(f"\n[JSON Parsing Error]: Model output was not valid JSON:\n{result.raw_text}\n")
        return {
            "status": "ERROR",
            "answer": "Failed to parse structured response from AI model.",
            "source_ids": [],
            "grounding": "UNSUPPORTED",
            "notes": f"JSON Decode Error: {str(json_err)}"
        }