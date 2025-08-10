import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import os
import fitz  # PyMuPDF
from langchain.schema import Document
from agent_core.rag.vectorstore import VectorDBHandler
from langchain_huggingface import HuggingFaceEmbeddings
import re
import numpy as np

# ----------------- CONFIG -----------------
PDF_DIR = Path(r"C:\Users\user\Documents\pdfs")
assert PDF_DIR.exists(), f"Directory not found: {PDF_DIR}"

DOCUMENT_METADATA = {
    "AutoGluon_Tabular_for_Credit_Risk_v2.pdf": {
        "title": "AutoGluonTabular",
        "source": "Documentation",
        "jurisdiction": "Global"
    },
    "Credit_Risk_ML_Fundamentals_v2.pdf": {
        "title": "Credit Risk ML Fundamentals",
        "source": "Documentation",
        "jurisdiction": "Global"
    }
}

# ----------------- SEMANTIC CHUNKER -----------------
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')

def _simple_sent_tokenize(text: str):
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    fused = []
    buf = ""
    for s in sents:
        if len(s.split()) < 4:
            buf = (buf + " " + s).strip()
            continue
        if buf:
            fused.append((buf + " " + s).strip())
            buf = ""
        else:
            fused.append(s)
    if buf:
        fused.append(buf)
    return fused

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)

def extract_text_chunks_semantic(
    file_path: str,
    embedder: HuggingFaceEmbeddings,
    chunk_token_target: int = 900,
    chunk_token_max: int = 1200,
    min_chunk_tokens: int = 300,
    overlap_sentences: int = 2,
    sim_threshold_merge: float = 0.58,
    sim_threshold_break_early: float = 0.42
):
    doc = fitz.open(file_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    if not full_text.strip():
        return []

    sents = _simple_sent_tokenize(full_text)
    if not sents:
        return []

    batch = 64
    sent_embs = []
    for i in range(0, len(sents), batch):
        vecs = embedder.embed_documents(sents[i:i+batch])
        sent_embs.extend([np.array(v, dtype=np.float32) for v in vecs])

    chunks = []
    i = 0
    while i < len(sents):
        cur_sents = [sents[i]]
        cur_tokens = _estimate_tokens(cur_sents[0])
        last_emb = sent_embs[i]
        i += 1

        while i < len(sents):
            next_sent = sents[i]
            next_emb = sent_embs[i]
            sim = _cos_sim(last_emb, next_emb)
            next_tokens = _estimate_tokens(next_sent)

            hard_cap_hit = (cur_tokens + next_tokens) > chunk_token_max
            soft_cap_hit = (cur_tokens + next_tokens) > chunk_token_target

            if hard_cap_hit:
                break

            if (not soft_cap_hit) or (sim >= sim_threshold_merge):
                cur_sents.append(next_sent)
                cur_tokens += next_tokens
                last_emb = next_emb
                i += 1
                continue

            if cur_tokens < min_chunk_tokens and sim >= sim_threshold_break_early:
                cur_sents.append(next_sent)
                cur_tokens += next_tokens
                last_emb = next_emb
                i += 1
                continue

            break

        chunk_text = " ".join(cur_sents).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if overlap_sentences > 0 and i < len(sents):
            i = max(i - overlap_sentences, 0)

        if i < len(sents) and chunks and chunks[-1].endswith(sents[i]):
            i += 1

    return chunks

# ----------------- INGESTION -----------------
def ingest_pdfs_to_vector_db():
    vector_db = VectorDBHandler()
    embedder = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    all_documents = []

    for file in PDF_DIR.glob("*.pdf"):
        base_name = file.name
        print(f"Processing {base_name}...")
        metadata = DOCUMENT_METADATA.get(base_name, {"title": base_name, "source": "Unknown", "jurisdiction": "Unknown"})

        try:
            chunks = extract_text_chunks_semantic(str(file), embedder=embedder)
            docs = [
                Document(page_content=chunk, metadata={**metadata, "filename": base_name})
                for chunk in chunks if chunk.strip()
            ]
            all_documents.extend(docs)
        except Exception as e:
            print(f"❌ Failed to process {base_name}: {e}")

    if all_documents:
        print(f"Ingesting {len(all_documents)} document chunks into FAISS...")
        vector_db.add_documents(all_documents)
        print("✅ Ingestion complete.")

if __name__ == "__main__":
    ingest_pdfs_to_vector_db()
