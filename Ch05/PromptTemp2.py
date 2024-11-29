from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv() # Load environment variables
# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")

template = ChatPromptTemplate.from_template("""
Human: What is the capital of {place}?
""")

# Generate the final prompt
input_variables = {"place": "California"}

# Send the prompt to the model and get the response
formatted_prompt = template.invoke(input=input_variables)
response = llm.invoke(formatted_prompt)
print(response.content)