import os
import re
import hashlib
from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def clean_text(text: str) -> str:
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

def extract_pdf(document_id: str, filename: str, filepath: str) -> Dict[str, Any]:
    reader = PdfReader(filepath)
    pages = []
    for idx, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        pages.append({"page_number": idx + 1, "text": raw})
    return {"document_id": document_id, "filename": filename, "pages": pages}

def extract_docx(document_id: str, filename: str, filepath: str) -> Dict[str, Any]:
    doc = Document(filepath)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return {"document_id": document_id, "filename": filename, "pages": [{"page_number": 1, "text": full_text}]}

def detect_sections(page_num: int, page_text: str) -> List[Dict[str, Any]]:
    # Simple regex detecting Section / Article / Heading boundaries
    pattern = r'(?P<section_title>(?:SECTION|ARTICLE|[0-9]+\.)\s+[A-Z0-9\s,\-\.\'\"]+)'
    matches = list(re.finditer(pattern, page_text, re.IGNORECASE))
    
    if not matches:
        return [{"page_number": page_num, "section": "General", "text": page_text}]
    
    sections = []
    for i, match in enumerate(matches):
        sec_title = match.group("section_title").strip()
        start_idx = match.end()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(page_text)
        sec_body = page_text[start_idx:end_idx].strip()
        sections.append({"page_number": page_num, "section": sec_title, "text": sec_body})
    return sections

def extract_clauses(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = section["text"]
    raw_clauses = re.split(r'\n(?=[0-9]+\.[0-9]+|\([a-z]\)|\([0-9]+\))', text)
    clauses = []
    
    for idx, c in enumerate(raw_clauses):
        c_clean = c.strip()
        if not c_clean:
            continue
        
        # Keyword category classification
        cat = "GENERAL"
        lower_c = c_clean.lower()
        if any(k in lower_c for k in ["salary", "pay", "compensation", "bonus"]):
            cat = "COMPENSATION"
        elif any(k in lower_c for k in ["terminate", "notice", "severance"]):
            cat = "TERMINATION"
        elif any(k in lower_c for k in ["confidential", "non-disclosure", "secret"]):
            cat = "CONFIDENTIALITY"
        elif any(k in lower_c for k in ["vacation", "leave", "holiday"]):
            cat = "LEAVE"
            
        clauses.append({
            "page": section["page_number"],
            "section": section["section"],
            "clause_number": f"C{idx+1}",
            "category": cat,
            "text": c_clean
        })
    return clauses

def chunk_clauses(clauses: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    chunk_records = []
    for clause in clauses:
        words = clause["text"].split()
        if not words:
            continue
            
        if len(words) <= chunk_size:
            chunk_records.append({**clause, "text": clause["text"]})
        else:
            start = 0
            sub_idx = 1
            while start < len(words):
                end = start + chunk_size
                chunk_text = " ".join(words[start:end])
                c_copy = clause.copy()
                c_copy["clause_number"] = f"{clause['clause_number']}-{sub_idx}"
                c_copy["text"] = chunk_text
                chunk_records.append(c_copy)
                sub_idx += 1
                start += (chunk_size - chunk_overlap)
    return chunk_records

def process_document_to_chunks(extraction: Dict[str, Any], chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    doc_id = extraction["document_id"]
    filename = extraction["filename"]
    all_clauses = []
    
    for page in extraction["pages"]:
        cleaned = clean_text(page["text"])
        if not cleaned:
            continue
        sections = detect_sections(page["page_number"], cleaned)
        for sec in sections:
            all_clauses.extend(extract_clauses(sec))
            
    raw_chunks = chunk_clauses(all_clauses, chunk_size, chunk_overlap)
    
    final_chunks = []
    for i, c in enumerate(raw_chunks, 1):
        chunk_id = f"{doc_id}-P{c['page']}-S1-{c['clause_number']}"
        final_chunks.append({
            "chunk_id": chunk_id,
            "document_id": doc_id,
            "filename": filename,
            "page": c["page"],
            "section": c["section"],
            "clause_number": c["clause_number"],
            "category": c["category"],
            "text": c["text"]
        })
    return final_chunks