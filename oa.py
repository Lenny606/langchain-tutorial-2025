import os
from getpass import getpass
from skimage import io
import matplotlib.pyplot as plt
from Paragraph import Paragraph
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate
)

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY') or getpass('OpenAI API Key: ')
openai_model = "gpt-4o-mini"

system_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant with name {name}.",
                                                          input_variables=["name"])
user_prompt = HumanMessagePromptTemplate.from_template(
    "Analyze human prompt and try to answer, prompt is: {human_input}", input_variables=["human_input"])

# to display template
# print(user_prompt.format(human_input="Hello, how are you?").content)
# merge user and system prompt
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])
# print(chat_prompt.format(human_input="Hello, how are you?", name="Tomik"))
#  0.0 most accurate, 1.0 most creative
llm = ChatOpenAI(temperature=0.5, model=openai_model)

# define simple Chain
chain_one = (
        {
            "human_input": lambda x: x["human_input"],
            "name": lambda x: x["name"]
        }
        | chat_prompt
        | llm
        | {"output_key": lambda x: x.content}
)
# run chain
response_one = chain_one.invoke({
    "human_input": "What is the day after tomorrow?",
    "name": "Tomik"
})

# --- second chain ---
system_prompt = SystemMessagePromptTemplate.from_template("You are SEO specialist")
second_user_prompt = HumanMessagePromptTemplate.from_template(
    "Analyze {output_key} and create SEO keywords for this topic", input_variables=["output_key"])

second_chat_prompt = ChatPromptTemplate.from_messages([system_prompt, second_user_prompt])

chain_two = (
        {
            "output_key": lambda x: x["output_key"]
        }
        | second_chat_prompt
        | llm
        | {"seo_keywords": lambda x: x.content}
)
response_two = chain_two.invoke({
    "output_key": response_one["output_key"],
})

# chain 3 -> Structured Output forces LLM to return structured data
structured_llm = llm.with_structured_output(Paragraph)

third_system_prompt = SystemMessagePromptTemplate.from_template("You are data analyst")
third_user_prompt = HumanMessagePromptTemplate.from_template(
    "Based on SEO keywords {seo_keywords}, create structured summary", input_variables=["seo_keywords"])

third_chat_prompt = ChatPromptTemplate.from_messages([third_system_prompt, third_user_prompt])

chain_three = (
        {
            "seo_keywords": lambda x: x["seo_keywords"]
        }
        | third_chat_prompt
        | structured_llm
        | {
            "original_paragraph": lambda x: x.original_paragraph,
            "edited_paragraph": lambda x: x.edited_paragraph,
            "feedback": lambda x: x.feedback,
        }
)

response_three = chain_three.invoke({
    "seo_keywords": response_two["seo_keywords"]
})

# image generation but first generate prompt for this image

image_prompt = PromptTemplate(
    input_variables=["seo_keywords"],
    template="Generate a prompt for an image of {seo_keywords} topic. less then 500 characters  "
)


def generate_and_show_image(image_prompt):
    image_url = DallEAPIWrapper().run(image_prompt)
    image_data = io.imread(image_url)

    plt.imshow(image_data)
    plt.axis('off')
    plt.show()


# wrap function in RunnableLambda
generate_image_runnable = RunnableLambda(generate_and_show_image)

# 4 chain
chain_four = (
    {
        "seo_keywords": lambda x: x["seo_keywords"]
    }
    | image_prompt
    | llm
    | (lambda x: x.content)
    | generate_image_runnable
)
chain_four.invoke({"seo_keywords": response_two["seo_keywords"]})