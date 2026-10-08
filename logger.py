import os
import csv
from typing import Dict, Any
from config import AUDIT_LOG_PATH, LOGS_DIR

def log_interaction(record: Dict[str, Any]):
    os.makedirs(LOGS_DIR, exist_ok=True)
    file_exists = os.path.exists(AUDIT_LOG_PATH)
    
    fieldnames = [
        "timestamp", "request_id", "query", "status", "retrieved_chunks",
        "best_similarity", "average_similarity", "grounding_status",
        "confidence", "gemini_status", "execution_attempts", "response_time_sec", "error"
    ]
    
    with open(AUDIT_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)