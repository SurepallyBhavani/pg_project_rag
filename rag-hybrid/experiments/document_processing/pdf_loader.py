import os
from dotenv import load_dotenv # type: ignore
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain.vectorstores import Chroma # type: ignore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # type: ignore
from langchain.chains import RetrievalQA # type: ignore

# Load environment variables from .env file
load_dotenv()

def load_pdf_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {pdf_path}")
    print("Sample page content:", documents[0].page_content[:500])
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    print("Sample chunk content:", chunks[0].page_content[:500])
    return chunks

def create_vector_db(chunks):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(chunks, embeddings)
    print("Vector database created with", len(chunks), "chunks.")
    return db

def build_qa_chain(db):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
    )
    return qa

if __name__ == "__main__":
    # Example usage: replace 'manual.pdf' with your PDF file
    pdf_file = "sample_manual_2pages.pdf"
    if os.path.exists(pdf_file):
        documents = load_pdf_documents(pdf_file)
        chunks = split_documents(documents)
        db = create_vector_db(chunks)
        qa = build_qa_chain(db)
        query = "What is the procedure for error code 404 in the manual?"
        result = qa.invoke(query)
        print("QA Result:", result)
    else:
        print(f"File not found: {pdf_file}")
