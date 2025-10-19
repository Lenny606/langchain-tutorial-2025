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
    FewShotChatMessagePromptTemplate,
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

# BASIC PROMPT
prompt = (
    "Answer the question. If you don't know the answer, just say that you don't know. -- Context: {context}")  # system prompt
prompt_template = ChatPromptTemplate.from_messages([
    ("system", prompt),
    ("user", "{query}")
])
# using templates
# prompt_template = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template(prompt),
#     HumanMessagePromptTemplate.from_template("{query}")
# ])

print(prompt_template.input_variables)
print(prompt_template.messages)

pipeline = prompt_template | llm

context = "Company Ford is a large automobile manufacturer in USA with rich history"
query = "What is the product of the company?"

# response = pipeline.invoke({"context": context, "query": query})  # feeds dictionary into prompt template

# another LCEL (LangChain Expression Language) , does the same thing
# pipeline = (
#         {
#             "context": lambda x: x["context"],
#             "query": lambda x: x["query"]
#         }
#         | prompt_template
#         | llm
# )

# print(response.content)

# FEW SHOTS PROMPT
example_prompt = ChatPromptTemplate.from_messages(
    [
        ('human', "{input}"),
        ("ai", "{output}")
    ]
)

examples = [
    {
        "input": "Query Number 1",
        "output": "Answer n 1"
    },
    {
        "input": "Query Number 3",
        "output": "Answer n 2"
    },
    {
        "input": "Query Number 3",
        "output": "Answer n 3"
    }
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples, example_prompt=example_prompt)
print(few_shot_prompt.format())

new_system_p = "Give summery of the context. Context: {context}"

prompt_template.messages[0].prompt.template = new_system_p
# output = pipeline.invoke({"context": context, "query": query}).content
# print(output)
# display(Markdown(out))

prompt_template_new = ChatPromptTemplate.from_messages([
    ('system', new_system_p),
    few_shot_prompt,
    ('user', "{query}")
])

pipeline_new = prompt_template_new | llm
# res = pipeline_new.invoke({"context": context, "query": query}).content
# print(res)
