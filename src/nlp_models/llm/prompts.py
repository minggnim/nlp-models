from typing import Any
from langchain import PromptTemplate


class ChatPrompt:
    def __init__(self) -> None:
        self.chat_template = """Assistant is a large language model.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {human_input}
Assistant:"""

        self.chat_prompt = PromptTemplate(input_variables=["history", "human_input"], template=self.chat_template)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.chat_prompt


class QAPrompt:
    def __init__(self) -> None:
        self.qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

        self.qa_prompt = PromptTemplate(input_variables=['context', 'question'], template=self.qa_template)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.qa_prompt
