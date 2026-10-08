import re
import json
from typing import List, Dict, Any, Tuple

def parse_model_json(raw_text: str) -> Dict[str, Any]:
    try:
        # Strip potential backtick codeblock wrappers
        cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        return {
            "ok": True,
            "answer": data.get("answer", ""),
            "source_ids": data.get("source_ids", []),
            "grounding": data.get("grounding", "UNSUPPORTED"),
            "notes": data.get("notes", "")
        }
    except Exception as e:
        return {"ok": False, "parse_error": str(e)}

def validate_sources(claimed_source_ids: List[str], retrieved_evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence_ids = {e["chunk_id"]: e for e in retrieved_evidence}
    confirmed_sources = []
    
    for sid in claimed_source_ids:
        if sid in evidence_ids:
            confirmed_sources.append(evidence_ids[sid])
            
    # Fallback to top retrieved evidence if model hallucinated citation IDs
    if not confirmed_sources and retrieved_evidence:
        confirmed_sources = retrieved_evidence
        
    return confirmed_sources

def compute_lexical_grounding(answer: str, validated_sources: List[Dict[str, Any]]) -> Tuple[str, float]:
    combined_evidence_text = " ".join([s["text"] for s in validated_sources]).lower()
    answer_words = set(re.findall(r'\w+', answer.lower())) - {"the", "a", "an", "is", "are", "to", "of", "in", "and", "or"}
    
    if not answer_words:
        return "FAIL", 0.0
        
    evidence_words = set(re.findall(r'\w+', combined_evidence_text))
    overlap = answer_words.intersection(evidence_words)
    ratio = len(overlap) / len(answer_words)
    
    if ratio >= 0.50:
        return "PASS", round(ratio, 4)
    elif ratio >= 0.30:
        return "REVIEW", round(ratio, 4)
    else:
        return "FAIL", round(ratio, 4)

def compute_confidence(validated_sources: List[Dict[str, Any]], grounding_status: str) -> Dict[str, Any]:
    if not validated_sources:
        return {"label": "LOW", "best_similarity": 0.0, "average_similarity": 0.0}
        
    sims = [s["similarity"] for s in validated_sources]
    best_sim = max(sims)
    avg_sim = sum(sims) / len(sims)
    
    if best_sim >= 0.75 and grounding_status == "PASS":
        label = "HIGH"
    elif best_sim >= 0.60 and grounding_status in ["PASS", "REVIEW"]:
        label = "MEDIUM"
    else:
        label = "LOW"
        
    return {
        "label": label,
        "best_similarity": round(best_sim, 4),
        "average_similarity": round(avg_sim, 4)
    }