from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from .base import LlmConfig


def build_vectordb(config: LlmConfig, file=None):
    if not file:
        loader = DirectoryLoader(config.DATA_PATH, glob='*.pdf', loader_cls=PyPDFium2Loader)
    else:
        loader = PyPDFium2Loader(file)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(config.FAISS_DB_PATH)


def load_vectordb(config: LlmConfig):
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local(config.FAISS_DB_PATH, embeddings)
    return vectordb
