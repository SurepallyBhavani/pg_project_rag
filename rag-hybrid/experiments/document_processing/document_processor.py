"""
Enhanced Document Processor
Extracts entities, relationships, and creates both vector and graph embeddings
"""

import re
import os
import hashlib
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime

# Try imports with fallbacks
try:
    import spacy # type: ignore
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not available. Entity extraction will use fallback method.")

try:
    import nltk # type: ignore
    from nltk.tokenize import sent_tokenize, word_tokenize # type: ignore
    from nltk.tag import pos_tag # type: ignore
    from nltk.chunk import ne_chunk # type: ignore
    from nltk.tree import Tree # type: ignore
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Will use basic text processing.")

try:
    from textblob import TextBlob # type: ignore
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not available. Sentiment analysis disabled.")

from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.embeddings import SentenceTransformerEmbeddings # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore

@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str
    document_id: str

@dataclass
class ExtractedRelationship:
    entity1: str
    entity2: str
    relationship_type: str
    confidence: float
    context: str
    document_id: str

@dataclass
class ProcessedDocument:
    document_id: str
    title: str
    content: str
    chunks: List[str]
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    concepts: List[Dict]
    metadata: Dict

class EnhancedDocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize embeddings
        try:
            self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            print("✓ Sentence transformer embeddings loaded")
        except Exception as e:
            print(f"⚠️  Sentence transformer failed, using fallback: {e}")
            # Use a simple TF-IDF based approach as fallback
            self.embeddings = None
        
        # Initialize NLP models
        self.nlp = None
        self._init_nlp_models()
        
        # Patterns for relationship extraction
        self.relationship_patterns = [
            # Causation patterns
            (r'(.+?)\s+(?:causes?|leads? to|results? in|triggers?)\s+(.+?)(?:\.|,|;|$)', 'CAUSES'),
            (r'(.+?)\s+(?:is caused by|results from|is triggered by)\s+(.+?)(?:\.|,|;|$)', 'CAUSED_BY'),
            
            # Definition patterns
            (r'(.+?)\s+(?:is|are|means?|refers? to|is defined as)\s+(.+?)(?:\.|,|;|$)', 'DEFINED_AS'),
            (r'(.+?):\s*(.+?)(?:\.|$)', 'DEFINED_AS'),
            
            # Comparison patterns
            (r'(.+?)\s+(?:is similar to|resembles|is like)\s+(.+?)(?:\.|,|;|$)', 'SIMILAR_TO'),
            (r'(.+?)\s+(?:differs from|is different from|contrasts with)\s+(.+?)(?:\.|,|;|$)', 'DIFFERENT_FROM'),
            
            # Hierarchy patterns
            (r'(.+?)\s+(?:is a type of|is a kind of|is a subset of)\s+(.+?)(?:\.|,|;|$)', 'SUBSET_OF'),
            (r'(.+?)\s+(?:includes?|contains?|comprises?)\s+(.+?)(?:\.|,|;|$)', 'INCLUDES'),
            
            # Association patterns
            (r'(.+?)\s+(?:is related to|is associated with|is connected to)\s+(.+?)(?:\.|,|;|$)', 'RELATED_TO'),
            (r'(.+?)\s+(?:and|with)\s+(.+?)\s+(?:are|work together)', 'ASSOCIATED_WITH'),
        ]
        
        # Academic/technical concept patterns
        self.concept_patterns = [
            r'(?:definition|def\.?):?\s*(.+?)(?:\n|\.)',
            r'(.+?)\s+is defined as\s+(.+?)(?:\.|$)',
            r'(.+?):\s*(.+?)(?:\.|$)',
            r'theorem\s+(.+?):\s*(.+?)(?:\.|$)',
            r'lemma\s+(.+?):\s*(.+?)(?:\.|$)',
            r'algorithm\s+(.+?):\s*(.+?)(?:\.|$)',
        ]
    
    def _init_nlp_models(self):
        """Initialize NLP models with fallback options"""
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("Loaded spaCy model for enhanced entity extraction")
            except OSError:
                print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('maxent_ne_chunker', quiet=True)
                nltk.download('words', quiet=True)
                print("NLTK models ready")
            except:
                print("NLTK download failed - some features may be limited")
    
    def process_pdf_documents(self, pdf_paths: List[str], graph_db_manager=None) -> Tuple[Any, List[ProcessedDocument]]:
        """
        Process multiple PDF documents and create both vector and graph embeddings
        """
        all_documents = []
        processed_docs = []
        
        # Load all PDFs
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Process each document
                for doc_index, doc in enumerate(documents):
                    doc_id = f"{os.path.basename(pdf_path)}_{doc_index}"
                    
                    # Extract content and metadata
                    content = doc.page_content
                    title = f"{os.path.basename(pdf_path)} - Page {doc_index + 1}"
                    metadata = doc.metadata
                    metadata.update({
                        'source_file': pdf_path,
                        'document_id': doc_id,
                        'processed_at': str(datetime.now())
                    })
                    
                    # Process the document
                    processed_doc = self._process_single_document(doc_id, title, content, metadata)
                    processed_docs.append(processed_doc)
                    
                    # Add to documents list for vector database
                    doc.metadata.update(metadata)
                    all_documents.append(doc)
                    
                    # Create graph embeddings if graph DB is available
                    if graph_db_manager:
                        self._create_graph_embeddings(processed_doc, graph_db_manager)
                
                print(f"Processed {len(documents)} pages from {pdf_path}")
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                continue
        
        # Create vector database
        if all_documents and self.embeddings:
            chunks = self.text_splitter.split_documents(all_documents)
            vector_db = Chroma.from_documents(chunks, self.embeddings)
            print(f"Created vector database with {len(chunks)} chunks")
        else:
            print("⚠️  Falling back to legacy processing due to embedding issues")
            vector_db = None
            chunks = []
        
        return vector_db, processed_docs
    
    def _process_single_document(self, doc_id: str, title: str, content: str, metadata: Dict) -> ProcessedDocument:
        """Process a single document and extract entities, relationships, concepts"""
        
        # Create chunks
        chunks = self._create_chunks(content)
        
        # Extract entities
        entities = self._extract_entities(content, doc_id)
        
        # Extract relationships
        relationships = self._extract_relationships(content, doc_id, entities)
        
        # Extract concepts
        concepts = self._extract_concepts(content, doc_id)
        
        return ProcessedDocument(
            document_id=doc_id,
            title=title,
            content=content,
            chunks=chunks,
            entities=entities,
            relationships=relationships,
            concepts=concepts,
            metadata=metadata
        )
    
    def _create_chunks(self, content: str) -> List[str]:
        """Split content into overlapping chunks"""
        # Use langchain text splitter
        chunks = self.text_splitter.split_text(content)
        return chunks
    
    def _extract_entities(self, content: str, doc_id: str) -> List[ExtractedEntity]:
        """Extract named entities using available NLP models"""
        entities = []
        
        if self.nlp and SPACY_AVAILABLE:
            # Use spaCy for entity extraction
            doc = self.nlp(content[:500])  # Limit content to first 500 chars for faster processing
            for ent in doc.ents:
                # Filter out very short or common entities
                if len(ent.text.strip()) > 2 and ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'EVENT', 'LAW', 'LANGUAGE']:
                    entities.append(ExtractedEntity(
                        name=ent.text.strip(),
                        entity_type=ent.label_,
                        confidence=0.9,  # spaCy doesn't provide confidence scores by default
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        context=content[max(0, ent.start_char-50):ent.end_char+50],
                        document_id=doc_id
                    ))
            print(f"🏷️  Extracted {len(entities)} entities from document {doc_id}")
        
        elif NLTK_AVAILABLE:
            # Use NLTK for entity extraction
            try:
                tokens = word_tokenize(content)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags)
                
                current_pos = 0
                for chunk in chunks:
                    if isinstance(chunk, Tree):
                        entity_name = ' '.join([token for token, pos in chunk.leaves()])
                        entity_type = chunk.label()
                        
                        # Find position in original text
                        start_pos = content.find(entity_name, current_pos)
                        if start_pos != -1:
                            end_pos = start_pos + len(entity_name)
                            entities.append(ExtractedEntity(
                                name=entity_name,
                                entity_type=entity_type,
                                confidence=0.8,
                                start_pos=start_pos,
                                end_pos=end_pos,
                                context=content[max(0, start_pos-50):end_pos+50],
                                document_id=doc_id
                            ))
                            current_pos = end_pos
            except Exception as e:
                print(f"NLTK entity extraction error: {e}")
        
        else:
            # Fallback: simple pattern-based entity extraction
            entities.extend(self._extract_entities_fallback(content, doc_id))
        
        # Add technical term extraction
        entities.extend(self._extract_technical_terms(content, doc_id))
        
        return entities
    
    def _extract_entities_fallback(self, content: str, doc_id: str) -> List[ExtractedEntity]:
        """Fallback entity extraction using simple patterns"""
        entities = []
        
        # Pattern for capitalized terms (potential proper nouns)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.finditer(pattern, content)
        
        for match in matches:
            name = match.group()
            # Skip common words
            if name.lower() not in ['The', 'This', 'That', 'When', 'Where', 'How', 'Why', 'What']:
                entities.append(ExtractedEntity(
                    name=name,
                    entity_type='ENTITY',
                    confidence=0.6,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=content[max(0, match.start()-50):match.end()+50],
                    document_id=doc_id
                ))
        
        return entities
    
    def _extract_technical_terms(self, content: str, doc_id: str) -> List[ExtractedEntity]:
        """Extract technical terms and acronyms"""
        entities = []
        
        # Pattern for acronyms (2-6 uppercase letters)
        acronym_pattern = r'\b[A-Z]{2,6}\b'
        matches = re.finditer(acronym_pattern, content)
        
        for match in matches:
            entities.append(ExtractedEntity(
                name=match.group(),
                entity_type='ACRONYM',
                confidence=0.8,
                start_pos=match.start(),
                end_pos=match.end(),
                context=content[max(0, match.start()-50):match.end()+50],
                document_id=doc_id
            ))
        
        # Pattern for technical terms (words ending with -tion, -ment, -ness, etc.)
        tech_pattern = r'\b\w+(?:tion|ment|ness|ism|ity|ing|algorithm|method|approach|technique|system|model|framework)\b'
        matches = re.finditer(tech_pattern, content, re.IGNORECASE)
        
        for match in matches:
            entities.append(ExtractedEntity(
                name=match.group(),
                entity_type='TECHNICAL_TERM',
                confidence=0.7,
                start_pos=match.start(),
                end_pos=match.end(),
                context=content[max(0, match.start()-50):match.end()+50],
                document_id=doc_id
            ))
        
        return entities
    
    def _extract_relationships(self, content: str, doc_id: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Extract relationships between entities using pattern matching"""
        relationships = []
        
        # Split content into sentences for better relationship extraction (limit to first 1000 chars)
        sentences = self._split_into_sentences(content[:1000])
        
        for sentence in sentences:
            # Apply relationship patterns
            for pattern, rel_type in self.relationship_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    entity1 = match.group(1).strip()
                    entity2 = match.group(2).strip()
                    
                    # Clean entities
                    entity1 = self._clean_entity_text(entity1)
                    entity2 = self._clean_entity_text(entity2)
                    
                    if entity1 and entity2 and entity1 != entity2:
                        relationships.append(ExtractedRelationship(
                            entity1=entity1,
                            entity2=entity2,
                            relationship_type=rel_type,
                            confidence=0.7,
                            context=sentence,
                            document_id=doc_id
                        ))
        
        # Extract co-occurrence relationships (entities in same sentence)
        cooccurrence_relationships = self._extract_cooccurrence_relationships(content, doc_id, entities)
        relationships.extend(cooccurrence_relationships)
        
        print(f"🔗 Extracted {len(relationships)} relationships from document {doc_id}")
        
        return relationships
    
    def _extract_cooccurrence_relationships(self, content: str, doc_id: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Extract relationships based on entity co-occurrence in sentences"""
        relationships = []
        sentences = self._split_into_sentences(content)
        
        for sentence in sentences:
            # Find entities in this sentence
            sentence_entities = []
            for entity in entities:
                if entity.name.lower() in sentence.lower():
                    sentence_entities.append(entity)
            
            # Create co-occurrence relationships
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    relationships.append(ExtractedRelationship(
                        entity1=entity1.name,
                        entity2=entity2.name,
                        relationship_type='CO_OCCURS_WITH',
                        confidence=0.5,
                        context=sentence,
                        document_id=doc_id
                    ))
        
        return relationships
    
    def _extract_concepts(self, content: str, doc_id: str) -> List[Dict]:
        """Extract concepts and their definitions"""
        concepts = []
        
        for pattern in self.concept_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(match.groups()) >= 2:
                    concept_name = match.group(1).strip()
                    definition = match.group(2).strip()
                else:
                    concept_name = match.group(1).strip()
                    definition = ""
                
                concept_name = self._clean_entity_text(concept_name)
                definition = self._clean_entity_text(definition)
                
                if concept_name and len(concept_name) > 2:
                    concepts.append({
                        'name': concept_name,
                        'definition': definition,
                        'document_id': doc_id,
                        'context': match.group(0),
                        'category': self._categorize_concept(concept_name, definition)
                    })
        
        return concepts
    
    def _create_graph_embeddings(self, processed_doc: ProcessedDocument, graph_db_manager):
        """Create graph embeddings for the processed document"""
        try:
            # Create document node
            doc_node_id = graph_db_manager.create_document_node(
                processed_doc.document_id,
                processed_doc.title,
                processed_doc.content[:1000],  # Truncate for storage
                processed_doc.metadata
            )
            
            # Create entity nodes
            entity_ids = {}
            for entity in processed_doc.entities:
                entity_id = graph_db_manager.create_entity_node(
                    entity.name,
                    entity.entity_type,
                    entity.context,
                    processed_doc.document_id
                )
                entity_ids[entity.name] = entity_id
            
            # Create concept nodes
            for concept in processed_doc.concepts:
                concept_id = graph_db_manager.create_concept_node(
                    concept['name'],
                    concept['definition'],
                    concept['category'],
                    processed_doc.document_id
                )
            
            # Create relationships
            for relationship in processed_doc.relationships:
                if relationship.entity1 in entity_ids and relationship.entity2 in entity_ids:
                    graph_db_manager.create_relationship(
                        entity_ids[relationship.entity1],
                        entity_ids[relationship.entity2],
                        relationship.relationship_type,
                        {'context': relationship.context},
                        relationship.confidence
                    )
            
            print(f"Created graph embeddings for {processed_doc.document_id}")
            
        except Exception as e:
            print(f"Error creating graph embeddings: {e}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using available tools"""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        # Fallback: simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _clean_entity_text(self, text: str) -> str:
        """Clean and normalize entity text"""
        if not text:
            return ""
        
        # Remove extra whitespace and punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)
        text = text.strip()
        
        # Limit length
        if len(text) > 100:
            text = text[:100] + "..."
        
        return text
    
    def _categorize_concept(self, name: str, definition: str) -> str:
        """Categorize concepts based on content"""
        name_lower = name.lower()
        def_lower = definition.lower()
        
        if any(word in name_lower for word in ['algorithm', 'method', 'approach', 'technique']):
            return 'METHOD'
        elif any(word in name_lower for word in ['theory', 'theorem', 'principle', 'law']):
            return 'THEORY'
        elif any(word in name_lower for word in ['system', 'framework', 'architecture', 'model']):
            return 'SYSTEM'
        elif any(word in def_lower for word in ['process', 'procedure', 'step']):
            return 'PROCESS'
        else:
            return 'CONCEPT'

# Example usage
if __name__ == "__main__":
    processor = EnhancedDocumentProcessor()
    
    # Test with a sample text
    sample_text = """
    Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. 
    
    Supervised learning algorithms learn from labeled training data, while unsupervised learning finds patterns in data without labels.
    
    Neural networks are computational models inspired by biological neural networks that constitute animal brains.
    """
    
    # Process sample document
    processed = processor._process_single_document("test_doc", "ML Sample", sample_text, {})
    
    print(f"Extracted {len(processed.entities)} entities:")
    for entity in processed.entities[:5]:
        print(f"  - {entity.name} ({entity.entity_type})")
    
    print(f"\nExtracted {len(processed.relationships)} relationships:")
    for rel in processed.relationships[:3]:
        print(f"  - {rel.entity1} -> {rel.relationship_type} -> {rel.entity2}")
    
    print(f"\nExtracted {len(processed.concepts)} concepts:")
    for concept in processed.concepts[:3]:
        print(f"  - {concept['name']}: {concept['definition'][:100]}")