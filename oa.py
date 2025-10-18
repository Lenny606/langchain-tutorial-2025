import os
from getpass import getpass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY') or getpass('OpenAI API Key: ')
openai_model = "gpt-4o-mini"

system_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
user_prompt = HumanMessagePromptTemplate.from_template(
    "Analyze human prompt and try to answer, prompt is: {human_input}", input_variables=["human_input"])

# to display template
# print(user_prompt.format(human_input="Hello, how are you?"))
# merge user and system prompt
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
print(chat_prompt.format(human_input="Hello, how are you?"))
#  0.0 most accurate, 1.0 most creative
llm = ChatOpenAI(temperature=0.5, model=openai_model)
