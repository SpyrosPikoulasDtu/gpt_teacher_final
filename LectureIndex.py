from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.embeddings import Embeddings
from langchain.docstore.document import Document

from typing import Type, Iterable, Optional, List

class LectureIndex(FAISS):
    """Wrapper around the FAISS VectorStore"""

    @classmethod
    def from_documents(cls, documents: List[Document], embedding: Embeddings):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
        docs_split = text_splitter.split_documents(documents)
        
        return FAISS.from_documents.__func__(cls, docs_split, embedding)

    def similarity_search_with_score_threshold(self, query: str, threshold: float):
        docs_and_scores = self.similarity_search_with_score(query)
        docs_and_scores = filter(lambda d_s : d_s[1] < threshold, docs_and_scores)
        docs = map(lambda d_s : d_s[0], docs_and_scores)
        return list(docs)
