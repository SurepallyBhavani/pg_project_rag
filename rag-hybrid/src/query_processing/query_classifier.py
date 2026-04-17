"""
Query Classifier Module
Classifies queries as simple (vector DB) or complex (graph DB) based on linguistic patterns
"""

import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os

class QueryClassifier:
    def __init__(self):
        self.nlp = None
        self.classifier = None
        self.is_trained = False
        
        # Complex query indicators
        self.complex_keywords = {
            'relationship': ['relationship', 'relation', 'connect', 'connection', 'link', 'between', 'among'],
            'comparison': ['compare', 'difference', 'versus', 'vs', 'similar', 'different', 'contrast'],
            'causation': ['cause', 'effect', 'because', 'due to', 'leads to', 'results in', 'impact'],
            'hierarchy': ['hierarchy', 'structure', 'organization', 'levels', 'categories', 'classification'],
            'multi_entity': ['and', 'or', 'both', 'either', 'multiple', 'several', 'all'],
            'temporal': ['before', 'after', 'during', 'while', 'sequence', 'order', 'timeline'],
            'conditional': ['if', 'when', 'unless', 'provided that', 'in case', 'depends on'],
            'complex_reasoning': ['analyze', 'evaluate', 'assess', 'determine', 'explain how', 'why does']
        }
        
        # Simple query indicators
        self.simple_keywords = {
            'definition': ['what is', 'define', 'definition of', 'meaning of', 'explain'],
            'factual': ['who', 'where', 'when', 'how much', 'how many'],
            'direct': ['list', 'show', 'find', 'get', 'give me', 'tell me'],
            'specific': ['specific', 'particular', 'exact', 'precise']
        }
        
        self._initialize_spacy()
        self._load_or_create_classifier()
    
    def _initialize_spacy(self):
        """Initialize spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _load_or_create_classifier(self):
        """Load existing classifier or create a new one with training data"""
        model_path = "models/query_classifier_model.pkl"
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                    self.is_trained = True
                    print("Loaded existing query classifier model")
            except:
                self._train_classifier()
        else:
            self._train_classifier()
    
    def _train_classifier(self):
        """Train the classifier with sample data"""
        # Training data for query classification
        training_data = [
            # Simple queries (0)
            ("What is machine learning?", 0),
            ("Define neural network", 0),
            ("Who invented the computer?", 0),
            ("When was Python created?", 0),
            ("List the types of algorithms", 0),
            ("Show me data structures", 0),
            ("What does API stand for?", 0),
            ("Explain recursion", 0),
            ("Find information about databases", 0),
            ("Tell me about programming languages", 0),
            ("How many types of sorting algorithms are there?", 0),
            ("What is the definition of polymorphism?", 0),
            ("Where is the main function located?", 0),
            ("Give me examples of design patterns", 0),
            ("Show the syntax for loops", 0),
            
            # Complex queries (1)
            ("How does machine learning relate to artificial intelligence?", 1),
            ("Compare supervised and unsupervised learning algorithms", 1),
            ("What is the relationship between data structures and algorithms?", 1),
            ("Analyze the impact of object-oriented programming on software development", 1),
            ("How do databases connect to web applications and what are the security implications?", 1),
            ("Explain the relationship between frontend and backend development", 1),
            ("What causes memory leaks and how do they affect system performance?", 1),
            ("Compare different software development methodologies and their effectiveness", 1),
            ("How do design patterns solve common programming problems and when should they be used?", 1),
            ("Analyze the connection between network protocols and web security", 1),
            ("What is the relationship between compiler optimization and runtime performance?", 1),
            ("How do microservices architecture compare to monolithic architecture in terms of scalability and maintenance?", 1),
            ("Explain how artificial intelligence and machine learning are transforming different industries", 1),
            ("What are the connections between data science, statistics, and machine learning?", 1),
            ("How do different programming paradigms influence software design and development?", 1),
        ]
        
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]
        
        # Create pipeline with TF-IDF and Naive Bayes
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', 
                ngram_range=(1, 3),
                max_features=1000
            )),
            ('nb', MultinomialNB())
        ])
        
        # Train the classifier
        self.classifier.fit(texts, labels)
        self.is_trained = True
        
        # Save the model
        try:
            with open("models/query_classifier_model.pkl", 'wb') as f:
                pickle.dump(self.classifier, f)
            print("Trained and saved new query classifier model")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def extract_entities(self, query):
        """Extract entities from query using spaCy"""
        if not self.nlp:
            return []
        
        doc = self.nlp(query)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            })
        
        return entities
    
    def analyze_query_complexity(self, query):
        """Analyze query complexity using keyword matching"""
        query_lower = query.lower()
        
        # Count complex indicators
        complex_score = 0
        simple_score = 0
        
        # Check for complex keywords
        for category, keywords in self.complex_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    complex_score += 1
        
        # Check for simple keywords
        for category, keywords in self.simple_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    simple_score += 1
        
        # Additional complexity indicators
        if len(query.split()) > 10:  # Long queries are often complex
            complex_score += 1
        
        if '?' in query and query.count('?') > 1:  # Multiple questions
            complex_score += 1
        
        if any(word in query_lower for word in ['and', 'or', 'but', 'however']):  # Conjunctions
            complex_score += 1
        
        return {
            'complex_score': complex_score,
            'simple_score': simple_score,
            'word_count': len(query.split()),
            'entities': self.extract_entities(query) if self.nlp else []
        }
    
    def classify_query(self, query):
        """
        Classify query as simple (0) or complex (1)
        Returns: dict with classification result and analysis
        """
        if not query or not query.strip():
            return {
                'classification': 'simple',
                'confidence': 0.0,
                'method': 'default',
                'analysis': {}
            }
        
        # Analyze query complexity
        analysis = self.analyze_query_complexity(query)
        
        # Rule-based classification as fallback
        rule_based_result = 'complex' if analysis['complex_score'] > analysis['simple_score'] else 'simple'
        
        # ML-based classification if model is available
        ml_result = rule_based_result
        ml_confidence = 0.5
        
        if self.is_trained and self.classifier:
            try:
                # Get prediction and probability
                prediction = self.classifier.predict([query])[0]
                probabilities = self.classifier.predict_proba([query])[0]
                
                ml_result = 'complex' if prediction == 1 else 'simple'
                ml_confidence = max(probabilities)
                
            except Exception as e:
                print(f"Error in ML classification: {e}")
                ml_result = rule_based_result
                ml_confidence = 0.5
        
        # Combine results (prioritize ML if confidence is high)
        final_result = ml_result if ml_confidence > 0.6 else rule_based_result
        
        return {
            'classification': final_result,
            'confidence': ml_confidence,
            'method': 'ml' if ml_confidence > 0.6 else 'rule-based',
            'analysis': analysis,
            'rule_based_result': rule_based_result,
            'ml_result': ml_result,
            'query_length': len(query.split()),
            'entities_found': len(analysis['entities'])
        }
    

    def get_classification_explanation(self, classification_result):
        """Generate human-readable explanation for classification"""
        result = classification_result['classification']
        method = classification_result['method']
        confidence = classification_result['confidence']
        analysis = classification_result['analysis']
        
        explanation = f"Query classified as **{result.upper()}** using {method} approach "
        explanation += f"(confidence: {confidence:.2f})\n\n"
        
        if result == 'complex':
            explanation += "**Reasoning for Complex Classification:**\n"
            explanation += f"- Complex indicators found: {analysis['complex_score']}\n"
            explanation += f"- Query length: {analysis['word_count']} words\n"
            if analysis['entities']:
                explanation += f"- Named entities detected: {len(analysis['entities'])}\n"
            explanation += "\n*This query will be processed using Graph Embeddings for better relationship understanding.*"
        else:
            explanation += "**Reasoning for Simple Classification:**\n"
            explanation += f"- Simple indicators found: {analysis['simple_score']}\n"
            explanation += f"- Query length: {analysis['word_count']} words\n"
            explanation += "\n*This query will be processed using Vector Embeddings for fast retrieval.*"
        
        return explanation

# Example usage and testing
if __name__ == "__main__":
    classifier = QueryClassifier()
    
    test_queries = [
        "What is machine learning?",
        "How does supervised learning relate to unsupervised learning and what are the key differences?",
        "Define neural network",
        "Compare different database management systems and explain their relationships",
        "List sorting algorithms",
        "Analyze the connection between frontend frameworks and backend APIs"
    ]
    
    print("=== Query Classification Test ===\n")
    for query in test_queries:
        result = classifier.classify_query(query)
        print(f"Query: {query}")
        print(f"Classification: {result['classification']} ({result['confidence']:.2f})")
        print(f"Method: {result['method']}")
        print("---")