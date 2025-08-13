from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
model = "sentence-transformers/all-MiniLM-L6-v2"

# Load raw PDF
# DirectoryLoader: Load from a directory and initialize with a path to directory and how to glob over it.
def load_document(folder_path):
    loader = DirectoryLoader(
        folder_path, # The folder that you want to load
        glob="*.pdf", # Find all file endswith ".pdf"
        loader_cls=PyPDFLoader
    )

    document = loader.load()
    return document

document = load_document(folder_path="data/")

# Create chunks
# RecursiveCharacterTextSplitter: là để chia nhỏ văn bản dài thành nhiều đoạn (chunk) sao cho mỗi đoạn không vượt quá giới hạn token/ký tự mà mô hình có thể xử lý.
def create_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # chunk_size: giới hạn văn bản cắt ra chỉ có 500 ký tự, chunk_overlap: cho phép các chunk sau bị chồng lấn với chunk trước bao nhiêu ký tự đó
    text_chunks = text_splitter.split_documents(document)
    return text_chunks

text_chunks = create_chunks(document)

# Create Vector Embeddings
def get_model(model):
    embedding_model = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    return embedding_model

embedding_model = get_model(model)

# Store embeddings in FAISS
db_faiss_path = "store/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(db_faiss_path)