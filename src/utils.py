import getpass
import os

from langchain_core.prompts import ChatPromptTemplate

# choose your favorite model # https://langchain-ai.github.io/langgraph/tutorials/workflows/#set-up
# maybe we want to set nvidia endpoints here too if we wanna use the NIMs
from langchain_ollama import ChatOllama

from LLMFineTuning.src._constants import LLM_CONFIGS

# def _set_env(var: str):
#     if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("LLAMA_API_KEY")  # get LLAMA api key?? need to download ChatOllama
# running Ollama is very different then OpenAI for that it would be like this:
# llm = ChatOpenAI(
#     model=LLM_CONFIGS["openai"]["model"], api_key=LLM_CONFIGS["openai"]["api_key"]
# )

# # but for Ollama we do this:
# llm = ChatOllama(model=LLM_CONFIGS["llama3.2"]["model"])
# messages = [
#     ("system", "You are a nice guy. Rewrite messages to make them sound nicer."),
#     (
#         "human",
#         "My friend told me I'm bossy. I want to tell them to get off this project.",
#     ),
# ]

# llm.invoke(messages)

# stream = llm.stream(messages)

# full = next(stream)
# for chunk in stream:
#     full += chunk
# print(full)


# json_llm = ChatOllama(format="json", model=LLM_CONFIGS["llama3.2"]["model"])
# messages = [
#     (
#         "human",
#         "Return a query for the weather in a random location and time of day with two keys: location and time_of_day. Respond using JSON only.",
#     ),
# ]
# print(llm.invoke(messages).content)

# but for Ollama we do this:
llm = ChatOllama(
    model=LLM_CONFIGS["llama3.2"]["model"],
    streaming=True,
)


# Start conversation history
messages = [("system", "You are a helpful assistant.")]

while True:
    user_input = input("\nYou: ")

    if user_input.lower() in {"exit", "quit"}:
        break

    # Add user message
    messages.append(("human", user_input))

    # Get response (streamed)
    response = llm.invoke(messages)
    print(response.content)

    # Add assistant message to history
    messages.append(response)
# response = llm.invoke("What is the capital of France?")

# messages = [
#     ("system", "You are a nice guy. Rewrite messages to make them sound nicer."),
#     (
#         "human",
#         "{input}",
#     ),
# ]

# llm.invoke(messages)
# for chunk in llm.stream(messages):
#     print(chunk.text(), end="")


# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that translates {input_language} to {output_language}.",
#         ),
#         ("human", "{input}"),
#     ]
# )

# chain = prompt | llm

# print(
#     chain.invoke(
#         {
#             "input_language": "English",
#             "output_language": "German",
#             "input": "I love programming.",
#         }
#     )
# )
