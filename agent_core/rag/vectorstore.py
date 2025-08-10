import os
from typing import List, Optional, Dict, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

FAISS_PATH = "agent_core/rag/faiss_index"
EMBED_MODEL = "intfloat/e5-large-v2"


class VectorDBHandler:
    def __init__(self):
        # E5: normalize embeddings is important
        self.embedder = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vstore: Optional[FAISS] = None
        self.bm25: Optional[BM25Retriever] = None
        self.hybrid: Optional[EnsembleRetriever] = None

        if os.path.exists(FAISS_PATH):
            self.vstore = FAISS.load_local(
                FAISS_PATH, self.embedder, allow_dangerous_deserialization=True
            )

        if self.vstore:
            all_docs = self._get_all_docs()
            if all_docs:
                self.bm25 = BM25Retriever.from_documents(all_docs)
                self.bm25.k = 4
                self.hybrid = EnsembleRetriever(
                    retrievers=[self.vstore.as_retriever(search_kwargs={"k": 4}), self.bm25],
                    weights=[0.6, 0.4],
                )

    # ---------- E5 helpers ----------
    @staticmethod
    def _prefix_passage(text: str) -> str:
        # E5 expects "passage: ..."
        return f"passage: {text}" if not text.startswith("passage: ") else text

    @staticmethod
    def _prefix_query(text: str) -> str:
        # E5 expects "query: ..."
        return f"query: {text}" if not text.startswith("query: ") else text

    # ---------- Docstore helpers ----------
    def _get_all_docs(self) -> List[Document]:
        """Safely get every Document in the FAISS docstore."""
        if not self.vstore:
            return []
        store = getattr(self.vstore, "docstore", None)
        if not store:
            return []
        # langchain InMemoryDocstore stores docs in _dict
        data = getattr(store, "_dict", None)
        if isinstance(data, dict):
            return list(data.values())
        # Fallback (won’t always work, but try)
        try:
            # This relies on private APIs; prefer _dict above
            return [store.search(i) for i in store._dict.keys()]
        except Exception:
            return []

    # ---------- Public API ----------
    def add_documents(self, docs: List[Document]):
        # E5: prefix passages before embedding
        prefixed_docs = [
            Document(page_content=self._prefix_passage(d.page_content), metadata=d.metadata)
            for d in docs
        ]

        if self.vstore is None:
            self.vstore = FAISS.from_documents(prefixed_docs, self.embedder)
        else:
            self.vstore.add_documents(prefixed_docs)

        self.vstore.save_local(FAISS_PATH)

        # Rebuild BM25 and hybrid
        all_docs = self._get_all_docs()
        if all_docs:
            self.bm25 = BM25Retriever.from_documents(all_docs)
            self.bm25.k = 4
            self.hybrid = EnsembleRetriever(
                retrievers=[self.vstore.as_retriever(search_kwargs={"k": 4}), self.bm25],
                weights=[0.6, 0.4],
            )

    def retrieve(
        self, query: str, k: int = 4, filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        query = self._prefix_query(query)
        results: List[Document] = []
        if self.hybrid:
            results = self.hybrid.get_relevant_documents(query)
        elif self.vstore:
            results = self.vstore.similarity_search(query, k=k)
        else:
            return []

        if filter_metadata:
            results = [
                d
                for d in results
                if all(d.metadata.get(key) == value for key, value in filter_metadata.items())
            ]
        return results[:k]

    def top1_with_score(self, query: str) -> Optional[Tuple[Document, float]]:
        """Return FAISS top-1 and its distance (lower is better)."""
        if not self.vstore:
            return None
        hits = self.vstore.similarity_search_with_score(self._prefix_query(query), k=1)
        if not hits:
            return None
        return hits[0]

    def add_section(self, section: str, content: str):
        self.add_documents([Document(page_content=content, metadata={"section": section})])

    def get_section(self, section: str, k: int = 4) -> Optional[str]:
        docs = self.retrieve(section, k=k, filter_metadata={"section": section})
        if not docs:
            return None
        return "\n\n".join(d.page_content for d in docs)
