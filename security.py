import re
from dataclasses import dataclass

@dataclass
class SecurityResult:
    is_suspicious: bool
    matched_pattern: str

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"bypass\s+security",
    r"override\s+your\s+rules",
    r"forget\s+instructions",
    r"act\s+as\s+an?\s+unrestricted"
]

def check_prompt_injection(query: str) -> SecurityResult:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return SecurityResult(is_suspicious=True, matched_pattern=pattern)
    return SecurityResult(is_suspicious=False, matched_pattern="")

def validate_query(query: str):
    q = query.strip()
    if not q:
        return type('Obj', (), {'is_valid': False, 'reason': 'Query cannot be empty.'})()
    if len(q) > 500:
        return type('Obj', (), {'is_valid': False, 'reason': 'Query exceeds maximum limit of 500 characters.'})()
    return type('Obj', (), {'is_valid': True, 'reason': 'Valid.'})()