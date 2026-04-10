from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
PDF_FILENAME = 'memory_langchain.pdf'


embeddings = OpenAIEmbeddings(
    model="qwen/qwen3-embedding-8b",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

def load_and_split_pdf(file_path: str):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = file_path

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(docs)
    
    # ensure page metadata exists
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks

def build_chroma_from_pdf(file_path: str, persist_dir: str = "./chroma_db"):
    chunks = load_and_split_pdf(file_path)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_collection",
        persist_directory=persist_dir,
    )

    vectorstore.persist()
    return vectorstore


if __name__ == '__main__':
    build_chroma_from_pdf(PDF_FILENAME)