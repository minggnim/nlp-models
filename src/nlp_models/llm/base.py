from dataclasses import dataclass


@dataclass
class LlmConfig:
    MODEL_BIN_PATH: str = '../models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q8_0.bin'
    MODEL_TYPE: str = 'llama'
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.01
    DATA_PATH: str = '../data/0_raw'
    CHUNK_SIZE: int = 200
    CHUNK_OVERLAP: int = 0
    FAISS_DB_PATH: str = '../vectordb'
    EMBEDDING_MODEL: str = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    RETURN_SOURCE_DOCUMENTS: bool = True
    VECTOR_COUNT: int = 2
