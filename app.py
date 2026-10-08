import os
import pandas as pd
import streamlit as st

from config import DOCS_DIR, AUDIT_LOG_PATH
from main import build_index_from_documents, answer_query

# Page Configuration
st.set_page_config(
    page_title="Legal Document Intelligence System",
    page_icon="⚖️",
    layout="wide"
)

# Initialize Session State
if "index" not in st.session_state:
    st.session_state.index = None

st.title("⚖️ Legal Document Intelligence System")
st.caption("Clause-aware contract analysis powered by Gemini & FAISS Hybrid Search")

# =====================================================================
# SIDEBAR: DOCUMENT UPLOAD & INDEXING ONLY
# =====================================================================
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload Contract PDF or DOCX files", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs(DOCS_DIR, exist_ok=True)
        for file in uploaded_files:
            file_path = os.path.join(DOCS_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to system.")

    if st.button("🔨 Build / Re-index Vector Store", use_container_width=True, type="primary"):
        with st.spinner("Processing documents and generating embeddings..."):
            st.session_state.index = build_index_from_documents(DOCS_DIR)
            st.success("Indexing completed successfully!")

# Auto-load existing index on initial launch if files exist
if st.session_state.index is None and os.path.exists(DOCS_DIR) and len(os.listdir(DOCS_DIR)) > 0:
    st.session_state.index = build_index_from_documents(DOCS_DIR)

# =====================================================================
# MAIN VIEW: Q&A, AUDIT LOGS, AND INDEXED FILES
# =====================================================================
tab1, tab2, tab3 = st.tabs(["💬 Contract Q&A", "📊 Audit Log Analysis", "📄 Indexed Files"])

# --- TAB 1: INTERACTIVE Q&A ---
with tab1:
    st.subheader("Ask Questions About Your Uploaded Contracts")
    
    user_query = st.text_input(
        "Enter your question:", 
        placeholder="e.g., What is the termination notice period?"
    )
    
    if st.button("Submit Query"):
        if st.session_state.index is None or not st.session_state.index.is_ready():
            st.warning("No document index found. Upload contract files and click 'Build / Re-index Vector Store'.")
        elif not user_query.strip():
            st.warning("Please type a valid question.")
        else:
            with st.spinner("Analyzing contracts..."):
                result = answer_query(st.session_state.index, user_query)

            status = result["status"]
            
            if status == "SUCCESS":
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("System Confidence", result["confidence"])
                with col2:
                    st.metric("Lexical Grounding", result["grounding"])

                st.markdown("### Answer")
                st.info(result["answer"])

                st.markdown("### Verified Source Evidence")
                for i, src in enumerate(result["sources"], 1):
                    with st.expander(f"Source {i}: {src['chunk_id']} (Similarity: {src['similarity']})"):
                        st.markdown(f"**Document:** `{src['document_name']}` | **Page:** `{src['page']}` | **Section:** `{src['section']}`")
                        st.markdown(f"**Clause:** `{src['clause_number']}`")
                        st.text_area("Chunk Content", src["text"], height=100, key=f"src_{i}", disabled=True)

            elif status == "BLOCKED":
                st.error("🚨 **Query Blocked**: Prompt injection pattern or safety policy violation detected.")
            elif status == "NO_EVIDENCE":
                st.warning("⚠️ **No Relevant Evidence Found**: The answer isn't present in the uploaded documents.")
            else:
                st.error(f"Error: {result.get('answer', 'Execution failed.')}")

# --- TAB 2: AUDIT LOG VIEWER ---
with tab2:
    st.subheader("System Audit Trail")
    if os.path.exists(AUDIT_LOG_PATH):
        audit_df = pd.read_csv(AUDIT_LOG_PATH)
        st.dataframe(audit_df, use_container_width=True)
        
        csv_data = audit_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Audit Log CSV",
            data=csv_data,
            file_name="audit_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No audit entries logged yet. Submit queries in the Q&A tab to populate this log.")

# --- TAB 3: INDEXED DOCUMENTS ---
with tab3:
    st.subheader("Indexed Document Metadata")
    if st.session_state.index and st.session_state.index.documents:
        st.write(f"**Total Indexed Files:** {len(st.session_state.index.documents)}")
        st.write(f"**Total Vector Store Chunks:** {len(st.session_state.index.chunk_metadata)}")
        
        docs_df = pd.DataFrame(st.session_state.index.documents)
        st.table(docs_df)
    else:
        st.info("No files currently indexed in the vector store.")