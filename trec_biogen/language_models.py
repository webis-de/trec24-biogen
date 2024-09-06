from os import environ
from typing import Literal, Mapping, TypeAlias

from dspy import LM as DSPyLanguageModel, OpenAI as DSPyOpenAI
from langchain_core.language_models import BaseLanguageModel as LangchainLanguageModel
from langchain_openai.chat_models import ChatOpenAI as LangchainChatOpenAI
from langchain_openai.llms import OpenAI as LangchainOpenAI
from pydantic.v1.types import SecretStr


LanguageModelName: TypeAlias = Literal[
    "blablador:Mistral-7B-Instruct-v0.3",
    "blablador:Mixtral-8x7B-Instruct-v0.1",
    "blablador:Llama3.1-8B-Instruct",
    "openai:gpt-4o-mini-2024-07-18",
]

_BLABLADOR_MODEL_NAMES: Mapping[LanguageModelName, str] = {
    "blablador:Mistral-7B-Instruct-v0.3": \
        " 1 - Mistral-7B-Instruct-v0.3 - the best option in general - fast and good (new version july 2024)",
    "blablador:Mixtral-8x7B-Instruct-v0.1": \
        "2 - Mixtral-8x7B-Instruct-v0.1 Slower with higher quality",
    "blablador:Llama3.1-8B-Instruct": \
        "6 - Llama3.1-8B-Instruct - A good model from META (update July 2024)",
}


def get_dspy_language_model(name: LanguageModelName) -> DSPyLanguageModel:
    if (
        name == "blablador:Mistral-7B-Instruct-v0.3" or
        name == "blablador:Mixtral-8x7B-Instruct-v0.1" or
        name == "blablador:GritLM-7B" or
        name == "blablador:starcoder2-15b" or
        name == "blablador:Llama3.1-8B-Instruct"
    ):
        blablador_name = _BLABLADOR_MODEL_NAMES[name]
        return DSPyOpenAI(
            model=blablador_name,
            api_key=environ["BLABLADOR_API_KEY"],
            api_base="https://helmholtz-blablador.fz-juelich.de:8000/v1/",
            stop=(
                "\n\n", 
                "---",
            ),
            temperature=0,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
        )
    elif (
        name == "openai:gpt-4o-mini-2024-07-18"
    ):
        return DSPyOpenAI(
            model=name.removeprefix("openai:"),
            api_key=environ["OPENAI_API_KEY"],
            api_base="https://api.openai.com/v1/",
            stop=(
                "\n\n", 
                "---",
            ),
            temperature=0,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
        )
    else:
        raise ValueError(f"Unknown language model: {name}")





def get_langchain_language_model(name: LanguageModelName) -> LangchainLanguageModel:
    if (
        name == "blablador:Mistral-7B-Instruct-v0.3" or
        name == "blablador:Mixtral-8x7B-Instruct-v0.1" or
        name == "blablador:GritLM-7B" or
        name == "blablador:starcoder2-15b" or
        name == "blablador:Llama3.1-8B-Instruct"
    ):
        blablador_name = _BLABLADOR_MODEL_NAMES[name]
        return LangchainOpenAI(
            model=blablador_name,
            api_key=SecretStr(environ["BLABLADOR_API_KEY"]),
            base_url="https://helmholtz-blablador.fz-juelich.de:8000/v1/",
            temperature=0,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
        )
    elif (
        name == "openai:gpt-4o-mini-2024-07-18"
    ):
        return LangchainChatOpenAI(
            model=name.removeprefix("openai:"),
            api_key=SecretStr(environ["OPENAI_API_KEY"]),
            base_url="https://api.openai.com/v1/",
            temperature=0,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
        )
    else:
        raise ValueError(f"Unknown language model: {name}")
