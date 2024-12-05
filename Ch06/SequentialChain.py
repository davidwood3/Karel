from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv() # Load environment variables

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

model=ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

# Chain 1: Generate a recipe
prompt1 =  ChatPromptTemplate.from_template("Write a detailed recipe for {dish}:")
chain1 = prompt1 | model
# Chain 2: Generate a shopping list
prompt2 =  ChatPromptTemplate.from_template("Generate a shopping list for {recipe}:")
chain2 = prompt2 | model

# Combine the chains
overall_chain = chain1 | chain2

# Run the chain
print(overall_chain.invoke({"dish": "spaghetti and meatballs"}).content)
