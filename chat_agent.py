from llama_index.tools import QueryEngineTool, ToolMetadata
import chainlit as cl
from chainlit.input_widget import Select, TextInput
import openai
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from llama_index.callbacks.base import CallbackManager
from index_wikipages import create_index
from utils import get_apikey
index = None


@cl.on_chat_start
async def on_chat_start():
    global index
    # Settings
    settings = await cl.ChatSettings(
        [
            Select(
                id="MODEL",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-4"],
                initial_index=0,
            ),
            TextInput(id="WikiPageRequest", label="Request Wikipage"),
        ]
    ).send()


def wikisearch_engine(index):
    query_engine = index.as_query_engine(
        response_mode="compact", verbose=True, similarity_top_k=10
    )
    return query_engine


def create_react_agent(MODEL):
    query_engine_tools = [
        QueryEngineTool(
            query_engine=wikisearch_engine(index),
            metadata=ToolMetadata(
                name="Wikipedia Search",
                description="Useful for performing searches on the wikipedia knowledgebase",
            ),
        )
    ]

    openai.api_key = get_apikey()
    llm = OpenAI(model=MODEL)
    agent = ReActAgent.from_tools(
        tools=query_engine_tools,
        llm=llm,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
        verbose=True,
    )
    return agent

@cl.on_settings_update
async def setup_agent(settings):
    global agent
    global index
    query = settings["WikiPageRequest"]
    index = create_index(query)

    print("on_settings_update", settings)
    MODEL = settings["MODEL"]
    agent = create_react_agent(MODEL)
    await cl.Message(
        author="Agent", content=f"""Wikipage(s) "{query}" successfully indexed"""
    ).send()


@cl.on_message
async def main(message: str):
    if agent:
        response = await cl.make_async(agent.chat)(message)
        await cl.Message(author="Agent", content=response).send()
