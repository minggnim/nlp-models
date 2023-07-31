from dataclasses import dataclass


@dataclass
class LlmConfig:
    MODEL_BIN_PATH: str = '../../models/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q8_0.bin'
    MODEL_TYPE: str = 'llama'
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.01
