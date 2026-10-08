from typing import Dict, Any, Optional, List
from indexer import ContractIndex
from security import validate_query, check_prompt_injection
from config import DEFAULT_TOP_K, DEFAULT_THRESHOLD, HYBRID_SEMANTIC_WEIGHT, HYBRID_KEYWORD_WEIGHT

def retrieve_evidence(
    index: ContractIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    method: str = "hybrid",
    document_filter: Optional[str] = None
) -> Dict[str, Any]:
    
    q_check = validate_query(query)
    if not q_check.is_valid:
        return {"status": "ERROR", "error": q_check.reason}
        
    sec_check = check_prompt_injection(query)
    if sec_check.is_suspicious:
        return {"status": "BLOCKED", "error": f"Pattern detected: {sec_check.matched_pattern}"}
        
    if not index.is_ready():
        return {"status": "ERROR", "error": "Index is empty. Upload documents first."}

    # Scope metadata filtering
    chunk_map = index.chunk_lookup()
    scoped_ids = set()
    for cid, c in chunk_map.items():
        if document_filter and document_filter != "All Documents" and c["filename"] != document_filter:
            continue
        scoped_ids.add(cid)

    if not scoped_ids:
        return {"status": "NO_EVIDENCE", "evidence": []}

    candidate_k = max(top_k * 3, 20)
    
    # 1. Semantic Search
    raw_semantic = index.search_semantic(query, candidate_k)
    semantic_matches = {cid: score for cid, score in raw_semantic if cid in scoped_ids}
    
    # 2. Keyword Search
    raw_keyword = index.search_keyword(query, candidate_k)
    keyword_matches = {cid: score for cid, score in raw_keyword if cid in scoped_ids}

    # 3. Hybrid Rank Fusion
    all_candidate_ids = set(semantic_matches.keys()).union(set(keyword_matches.keys()))
    fused_scores = {}
    
    for cid in all_candidate_ids:
        sem_s = semantic_matches.get(cid, 0.0)
        kw_s = keyword_matches.get(cid, 0.0)
        
        if method == "hybrid":
            final_score = (HYBRID_SEMANTIC_WEIGHT * sem_s) + (HYBRID_KEYWORD_WEIGHT * kw_s)
        elif method == "semantic":
            final_score = sem_s
        else:
            final_score = kw_s
            
        fused_scores[cid] = final_score

    # 4. Threshold Gate Filtering
    sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    above_threshold = [(cid, score) for cid, score in sorted_candidates if score >= threshold]

    if not above_threshold:
        return {"status": "NO_EVIDENCE", "evidence": []}

    # Select top_k evidence records
    selected_evidence = []
    for cid, score in above_threshold[:top_k]:
        meta = chunk_map[cid]
        selected_evidence.append({
            "chunk_id": cid,
            "document_name": meta["filename"],
            "page": meta["page"],
            "section": meta["section"],
            "clause_number": meta["clause_number"],
            "similarity": round(score, 4),
            "text": meta["text"]
        })

    return {"status": "HAS_EVIDENCE", "evidence": selected_evidence}