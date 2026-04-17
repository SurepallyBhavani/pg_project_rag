"""
Configuration settings for the Hybrid RAG System
"""

import os

class Config:
    """Configuration class for the Hybrid RAG System"""
    
    # Database configurations
    VECTOR_DB_PATH = "chroma_db"
    GRAPH_DB_URI = "bolt://localhost:7687"
    GRAPH_DB_USER = "neo4j"
    GRAPH_DB_PASSWORD = "password"
    
    # Model configurations
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SPACY_MODEL = "en_core_web_sm"
    
    # Document processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Query processing
    SIMILARITY_THRESHOLD = 0.1
    MAX_RESULTS = 5
    
    # Data paths
    DATA_FOLDER = "data"
    MODEL_FOLDER = "models"
    
    # Flask configuration
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5001
    FLASK_DEBUG = True
    
    # AI API configuration
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-your-api-key-here")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    
    # NLP configurations
    ENTITY_EXTRACTION_BATCH_SIZE = 100
    MIN_ENTITY_LENGTH = 2
    MAX_RELATIONSHIPS_PER_DOCUMENT = 50
    
    @classmethod
    def get_template_path(cls):
        """Get the path to templates folder"""
        return os.path.join("src", "web_interface", "templates")
    
    @classmethod
    def get_model_path(cls, model_name):
        """Get the path to a specific model file"""
        return os.path.join(cls.MODEL_FOLDER, model_name)