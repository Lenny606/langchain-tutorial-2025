import os
from getpass import getpass

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY') or getpass('OpenAI API Key: ')

openai_model = "gpt-4o-mini"