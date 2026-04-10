from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

from dotenv import load_dotenv
import os

from config import config

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

embeddings = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    base_url=config.BASE_URL,
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

def build_chroma_from_pdf(file_path: str):
    chunks = load_and_split_pdf(file_path)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    vectorstore.persist()
    return vectorstore


if __name__ == '__main__':
    build_chroma_from_pdf(config.PDF_FILENAME)