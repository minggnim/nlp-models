import click
from nlp_models.llm.base import LlmConfig
from nlp_models.llm.llms import build_llm
from nlp_models.llm.apps import ChatLlmApp


config = LlmConfig(
    MODEL_BIN_PATH='./models/Llama-2-7B-Chat-GGML',
    DATA_PATH='./data/0_raw',
)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--inputs', '-s', type=str, default='Hi', help="start your conversation here")
def chat(inputs):
    llm_chat_app = ChatLlmApp(llm=build_llm(config), verbose=False)
    print(llm_chat_app(inputs))


if __name__ == '__main__':
    cli()
