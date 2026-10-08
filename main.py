import os
import time
import uuid
from typing import Dict, Any, Optional

from config import DOCS_DIR, DEFAULT_TOP_K, DEFAULT_THRESHOLD
from document_processor import extract_pdf, extract_docx, process_document_to_chunks, compute_file_hash
from indexer import ContractIndex
from retriever import retrieve_evidence
from generator import build_context_block, generate_answer_with_backoff
from validator import parse_model_json, validate_sources, compute_lexical_grounding, compute_confidence
from logger import log_interaction

def build_index_from_documents(doc_dir: str = DOCS_DIR) -> ContractIndex:
    index = ContractIndex()
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir, exist_ok=True)
        return index

    files = [f for f in os.listdir(doc_dir) if f.lower().endswith(('.pdf', '.docx'))]
    all_chunks = []
    
    for idx, fname in enumerate(files, 1):
        fpath = os.path.join(doc_dir, fname)
        with open(fpath, "rb") as f:
            data = f.read()
        fhash = compute_file_hash(data)
        doc_id = f"DOC{idx:03d}"
        
        if fname.lower().endswith('.pdf'):
            ext = extract_pdf(doc_id, fname, fpath)
        else:
            ext = extract_docx(doc_id, fname, fpath)
            
        chunks = process_document_to_chunks(ext)
        all_chunks.extend(chunks)
        index.documents.append({"document_id": doc_id, "filename": fname, "file_hash": fhash})
        index.known_hashes.add(fhash)

    if all_chunks:
        index.add_chunks(all_chunks)
        index.save_to_disk()
        print(f"Indexed {len(files)} files successfully ({len(all_chunks)} chunks total).")
    return index

def answer_query(
    index: ContractIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    document_filter: Optional[str] = None
) -> Dict[str, Any]:
    
    start_time = time.time()
    req_id = str(uuid.uuid4())[:8]

    # 1. Hybrid Retrieval & Security Checks
    retrieval = retrieve_evidence(index, query, top_k=top_k, threshold=threshold, document_filter=document_filter)
    
    if retrieval["status"] in ["ERROR", "BLOCKED", "NO_EVIDENCE"]:
        elapsed = round(time.time() - start_time, 3)
        log_interaction({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": req_id,
            "query": query,
            "status": retrieval["status"],
            "retrieved_chunks": 0,
            "best_similarity": 0.0,
            "average_similarity": 0.0,
            "grounding_status": "N/A",
            "confidence": "LOW",
            "gemini_status": "SKIPPED",
            "execution_attempts": 0,
            "response_time_sec": elapsed,
            "error": retrieval.get("error", "No context cleared threshold gate.")
        })
        
        msg = retrieval.get("error") or "The requested information was not found with sufficient confidence in the uploaded contracts."
        return {"status": retrieval["status"], "answer": msg, "confidence": "LOW", "grounding": "N/A", "sources": []}

    evidence = retrieval["evidence"]
    context_block = build_context_block(evidence)
    
    # 2. Gemini Generation Call
    gen_res = generate_answer_with_backoff(query, context_block)
    
    if not gen_res.success:
        elapsed = round(time.time() - start_time, 3)
        log_interaction({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": req_id, "query": query, "status": "ERROR", "retrieved_chunks": len(evidence),
            "best_similarity": max(e["similarity"] for e in evidence), "average_similarity": round(sum(e["similarity"] for e in evidence)/len(evidence), 4),
            "grounding_status": "N/A", "confidence": "LOW", "gemini_status": "FAILED",
            "execution_attempts": gen_res.attempts, "response_time_sec": elapsed, "error": gen_res.error
        })
        return {"status": "ERROR", "answer": "Model generation failed.", "confidence": "LOW", "sources": []}

    # 3. Output Validation
    parsed = parse_model_json(gen_res.raw_text)
    if not parsed["ok"]:
        return {"status": "ERROR", "answer": "Response parsing failure.", "confidence": "LOW", "sources": []}

    validated_sources = validate_sources(parsed["source_ids"], evidence)
    grounding_status, _ = compute_lexical_grounding(parsed["answer"], validated_sources)
    conf_dict = compute_confidence(validated_sources, grounding_status)
    elapsed = round(time.time() - start_time, 3)

    log_interaction({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "request_id": req_id, "query": query, "status": "SUCCESS", "retrieved_chunks": len(validated_sources),
        "best_similarity": conf_dict["best_similarity"], "average_similarity": conf_dict["average_similarity"],
        "grounding_status": grounding_status, "confidence": conf_dict["label"], "gemini_status": "SUCCESS",
        "execution_attempts": gen_res.attempts, "response_time_sec": elapsed, "error": ""
    })

    return {
        "status": "SUCCESS",
        "answer": parsed["answer"],
        "confidence": conf_dict["label"],
        "grounding": grounding_status,
        "sources": validated_sources
    }

if __name__ == "__main__":
    print("Building Document Index...")
    index = build_index_from_documents()
    
    if not index.is_ready():
        print(f"Please place your contract files (.pdf or .docx) inside '{DOCS_DIR}' and re-run.")
    else:
        print("\n--- Legal Document Assistant Ready ---")
        q = "What is the termination notice period?"
        print(f"\nQuery: {q}")
        res = answer_query(index, q)
        print(f"Answer: {res['answer']}")
        print(f"Confidence: {res['confidence']} | Grounding: {res['grounding']}")
        print("Sources:", [s["chunk_id"] for s in res["sources"]])