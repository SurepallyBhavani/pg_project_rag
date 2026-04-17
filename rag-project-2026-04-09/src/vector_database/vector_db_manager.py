"""
Vector Database Manager
Handles ChromaDB operations for document embeddings and similarity search
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

@dataclass
class VectorSearchResult:
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    document_name: str
    page_number: int

class VectorDatabaseManager:
    """Manages vector database operations using ChromaDB"""
    
    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
        self.collection = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def initialize(self):
        """Initialize the vector database components"""
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                # Initialize embeddings
                self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                print("[ok] Sentence transformer embeddings loaded")
                
                # Initialize ChromaDB
                if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                    # Load existing database
                    self.vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings
                    )
                    print(f"[ok] Loaded existing vector database from {self.persist_directory}")
                else:
                    # Create new database
                    self.vectorstore = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings
                    )
                    print("[ok] Created new vector database")
                    
                return True
            
        except Exception as e:
            print(f"Error initializing vector database: {e}")
            return False
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector database"""
        try:
            if self.vectorstore is None:
                raise ValueError("Vector database not initialized")
                
            # Create text chunks
            all_texts = []
            all_metadatas = []
            
            for text, metadata in zip(texts, metadatas):
                chunks = self.text_splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_id'] = i
                    chunk_metadata['total_chunks'] = len(chunks)
                    all_texts.append(chunk)
                    all_metadatas.append(chunk_metadata)
            
            # Add to vector store in batches to stay under Chroma client limits.
            batch_size = 4000
            for start_index in range(0, len(all_texts), batch_size):
                end_index = start_index + batch_size
                self.vectorstore.add_texts(
                    texts=all_texts[start_index:end_index],
                    metadatas=all_metadatas[start_index:end_index]
                )
            self.vectorstore.persist()
            
            print(f"[ok] Added {len(all_texts)} text chunks to vector database")
            return True
            
        except Exception as e:
            print(f"Error adding documents to vector database: {e}")
            return False

    def replace_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> bool:
        """Replace the current persistent store with a fresh ingest."""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)

        self.vectorstore = None
        if not self.initialize():
            return False

        return self.add_documents(texts, metadatas)

    def as_langchain_store(self):
        """Expose the underlying store for app compatibility."""
        return self.vectorstore
    
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = 0.1) -> List[VectorSearchResult]:
        """Perform similarity search in vector database"""
        try:
            if self.vectorstore is None:
                raise ValueError("Vector database not initialized")
            
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Filter by score threshold and format results
            filtered_results = []
            for doc, score in results:
                if score <= score_threshold:  # Lower score = higher similarity
                    result = VectorSearchResult(
                        content=doc.page_content,
                        metadata=doc.metadata,
                        similarity_score=1 - score,  # Convert to similarity (higher = more similar)
                        document_name=doc.metadata.get('source', 'Unknown'),
                        page_number=doc.metadata.get('page', 0)
                    )
                    filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []
    
    def keyword_search(self, query: str, keywords: List[str], k: int = 5) -> List[VectorSearchResult]:
        """Enhanced search that considers both similarity and keyword matching"""
        try:
            # First get similarity results
            similarity_results = self.similarity_search(query, k=k*2)  # Get more results initially
            
            # Score based on keyword presence
            scored_results = []
            for result in similarity_results:
                keyword_score = 0
                content_lower = result.content.lower()
                
                for keyword in keywords:
                    if keyword.lower() in content_lower:
                        keyword_score += 1
                
                # Combine similarity and keyword scores
                if keyword_score > 0:
                    combined_score = result.similarity_score * 0.7 + (keyword_score / len(keywords)) * 0.3
                    result.similarity_score = combined_score
                    scored_results.append(result)
            
            # Sort by combined score and return top k
            scored_results.sort(key=lambda x: x.similarity_score, reverse=True)
            return scored_results[:k]
            
        except Exception as e:
            print(f"Error performing keyword search: {e}")
            return self.similarity_search(query, k)  # Fallback to regular search
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        try:
            if self.vectorstore is None:
                return {"status": "not_initialized"}
            
            # Get collection info
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = client.list_collections()
            
            if collections:
                collection = collections[0]
                count = collection.count()
                
                return {
                    "status": "active",
                    "total_documents": count,
                    "collection_name": collection.name,
                    "embedding_model": "all-MiniLM-L6-v2"
                }
            else:
                return {
                    "status": "empty",
                    "total_documents": 0
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def clear_database(self) -> bool:
        """Clear all documents from the vector database"""
        try:
            if self.vectorstore is not None:
                # Delete the persist directory
                import shutil
                if os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory)
                
                # Reinitialize
                self.initialize()
                return True
                
        except Exception as e:
            print(f"Error clearing vector database: {e}")
            return False
