# 🚀 How to Run the Hybrid RAG Project

## 📋 Prerequisites
- Python 3.8 or higher
- Virtual environment activated

## ⚡ Quick Start Commands

### 1. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 2. Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

### 3. Add Your Documents
Place your PDF files in the `data/` folder:
```bash
# Syllabus PDFs go in:
data/syllabus/

# Notes PDFs go in:
data/notes/
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Interface
Open your browser and go to:
```
http://localhost:5001
```

## 🧪 Run Tests (Optional)
To validate the system's accuracy:
```bash
# Make sure the app is running first, then in a new terminal:
python test_rag_accuracy.py
```

## 📊 Check Ground Truth Responses
To see expected correct responses:
```bash
python ground_truth_responses.py
```

## 🛑 Stop the Application
Press `CTRL+C` in the terminal running the app

---

## 📁 Project Structure
```
├── app.py                      # Main application
├── requirements.txt            # Python dependencies
├── data/                       # PDF documents folder
├── src/                        # Source code modules
│   ├── vector_database/        # ChromaDB operations
│   ├── graph_database/         # Neo4j/NetworkX operations
│   ├── query_processing/       # ML query classification
│   ├── document_processing/    # PDF & NLP processing
│   └── web_interface/          # ChatGPT-style UI
├── models/                     # Trained ML models
├── config/                     # Configuration settings
├── test_rag_accuracy.py        # Test suite
└── ground_truth_responses.py   # Expected responses

```

## 💡 Tips
- The system auto-processes documents on startup
- First run may take longer (downloading ML models)
- Vector database is stored in `chroma_db/` folder
- System works offline after initial model downloads

## ⚠️ Troubleshooting
- **Port 5001 already in use**: Change port in `app.py` or stop other Flask apps
- **Neo4j connection refused**: This is normal - system uses NetworkX fallback
- **Slow first run**: ML models are downloading (one-time only)
