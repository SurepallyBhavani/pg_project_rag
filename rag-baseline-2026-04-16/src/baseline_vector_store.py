from __future__ import annotations

import os
import shutil
from typing import Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma


class BaselineVectorStore:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def initialize(self) -> bool:
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
            else:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
            return True
        except Exception as exc:
            print(f"Baseline vector store init failed: {exc}")
            return False

    def rebuild(self, texts: List[str], metadatas: List[Dict[str, str]]) -> bool:
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        self.vectorstore = None
        if not self.initialize():
            return False

        all_texts = []
        all_metadatas = []
        for text, metadata in zip(texts, metadatas):
            chunks = self.text_splitter.split_text(text)
            for index, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = str(index)
                all_texts.append(chunk)
                all_metadatas.append(chunk_metadata)

        batch_size = 4000
        for start_index in range(0, len(all_texts), batch_size):
            end_index = start_index + batch_size
            self.vectorstore.add_texts(
                texts=all_texts[start_index:end_index],
                metadatas=all_metadatas[start_index:end_index],
            )
        self.vectorstore.persist()
        return True

    def search(self, query: str, k: int = 4):
        return self.vectorstore.similarity_search_with_score(query, k=k)
