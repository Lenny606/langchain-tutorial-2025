import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langsmith import traceable

from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder
)

load_dotenv()
# setup for langsmith for tracing
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT'] = "default"
openai_model = "gpt-4o-mini"

llm = ChatOpenAI(temperature=0.0, model=openai_model)

# CONVERSATION BUFFER MEMORY with RunnableMessageHistory
# simplest, list of stored messages
conversation_buffer = [
    "system: You are a helpful AI assistant.",
    "human: Hello! How can I help you today?",
    "assistant: I'm here to assist you with any questions or tasks you may have."
]

system_prompt = "You are a helpful AI assistant."

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{query}"),
])

pipeline = prompt | llm

chat_map = {}
@traceable(name="Get chat history")
def get_chat_history(sessionId: str) -> InMemoryChatMessageHistory:
    """
    Retrieves the chat history associated with a given session ID. If the
    session ID does not exist in the mapping, a new chat history will be
    created and associated with the given session ID.

    :param sessionId: A unique identifier for the chat session.
    :type sessionId: str
    :return: The chat message history associated with the specified session
        ID.
    :rtype: InMemoryChatMessageHistory
    """
    if sessionId not in chat_map:
        # if id not in map, create new chat history
        chat_map[sessionId] = InMemoryChatMessageHistory()
    return chat_map[sessionId]

pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key='query',
    history_messages_key='history'
)

pipeline_with_history.invoke(
    {"query": "Hello. What today day of the week?"},
    config={"session_id": "id_5515464"}
    )
pipeline_with_history.invoke(
    {"query": "And what is the weather like?"},
    config={"session_id": "id_5515464"}
    )

pipeline_with_history.invoke(
    {"query": "What was the day again?"},
    config={"session_id": "id_5515464"}
    )

#  CONVERSATION BUFFER WINDOW MEMORY with RunnableMessageHistory
# keeps track of the last k messages, (less tokens are send, more tokens can be worse for llm, there is context limit )
