"""
To run Llama2 chat UI on CPU
1. Install the latest nlp_models package `pip install -U nlp_models`
2. Download `llama-2-7b-chat.ggmlv3.q8_0.bin` from https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main and place it under `models` folder
3. Run `streamlit run app.py` to spin up the UI
"""


import streamlit as st
from langchain.llms import CTransformers
from langchain import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from nlp_models.llm.base import LlmConfig
from nlp_models.llm.prompts import ChatPrompt


st.set_page_config(page_title='🤖 Llama2 chatbot on CPU', layout='wide', page_icon='🤖')


if 'input' not in st.session_state:
    st.session_state['input'] = ''
if 'temp' not in st.session_state:
    st.session_state['temp'] = ''
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'stored_session' not in st.session_state:
    st.session_state['stored_session'] = []


def clear_input():
    st.session_state.temp = st.session_state.input
    st.session_state.input = ""


def get_user_input():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input(
        "You: ",
        st.session_state["input"],
        key="input",
        placeholder="Ask me anything ...",
        on_change=clear_input,
        label_visibility='hidden'
    )
    input_text = st.session_state.temp
    return input_text


def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ''
    st.session_state['temp'] = ''
    st.session_state.memory = ConversationBufferWindowMemory()


with st.sidebar:
    st.markdown("---")
    st.markdown("# About")
    st.markdown("🦙 Llama2 🤖 Chatbot on CPU")
    st.markdown("The whole internet is within 8G of your laptop's memory")
    st.markdown("How cool is that!")
    st.markdown("---")


# Set up the Streamlit app layout
st.title("🦙 Llama2 🤖 Chatbot on CPU")
st.subheader(" Powered by 🦙 Llama2 + 🦜 LangChain + Streamlit")


cfg = LlmConfig()
llm = CTransformers(
    model=cfg.MODEL_BIN_PATH,
    model_type=cfg.MODEL_TYPE,
    config={
        'max_new_tokens': cfg.MAX_NEW_TOKENS,
        'temperature': cfg.TEMPERATURE
    }
)

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory()

chat_chain = LLMChain(
    llm=llm,
    prompt=ChatPrompt().chat_prompt,
    verbose=True,
    memory=st.session_state.memory,
)

st.sidebar.button("New Chat", on_click=new_chat, type='primary')

user_input = get_user_input()

if user_input:
    output = chat_chain.predict(human_input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i], icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    download_str = '\n'.join(download_str)

    if download_str:
        st.download_button('Download', download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.sidebar.checkbox("Clear-all"):
    if st.session_state.stored_session:
        st.session_state.stored_session = []
