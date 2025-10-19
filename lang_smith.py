import os
from dotenv import load_dotenv
import time
import random
from tqdm.auto import tqdm
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langchain_core.runnables import RunnableLambda
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate
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


# llm.invoke("Hello")

# decorator to track not lanchain functions
@traceable(name="Random number generator")
def generate_random_number():
    return random.randint(1, 10)


@traceable(name="Delay generator")
def generate_delay():
    delay = generate_random_number()
    time.sleep(delay)
    return 1


@traceable(name="Error generator")
def generate_random_error():
    number = generate_random_number()
    if number % 2 == 0:
        raise ValueError("Number error")
    return "No error"


for _ in tqdm(range(10)):
    generate_delay()
    try:
        generate_random_error()
    except ValueError:
        pass
