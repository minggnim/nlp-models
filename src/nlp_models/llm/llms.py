from langchain.llms import CTransformers
from langchain import LLMChain
from .base import LlmConfig
from .prompts import ChatPrompt


cfg = LlmConfig()
llm = CTransformers(
    model=cfg.MODEL_BIN_PATH,
    model_type=cfg.MODEL_TYPE,
    config={
        'max_new_tokens': cfg.MAX_NEW_TOKENS,
        'temperature': cfg.TEMPERATURE
    }
)

chat_chain = LLMChain(
    llm=llm,
    prompt=ChatPrompt().chat_prompt,
    verbose=True,
    memory=st.session_state.memory,
)
