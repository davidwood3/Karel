from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)

chain = ConversationChain(
    llm=llm,
    verbose=True, #k=? is the number of previous messages to remember
    memory=ConversationSummaryBufferMemory (llm=llm, max_token_limit=5, return_messages=True))


chain.invoke(
input="I am writing a book about mathematics, can you explain what is mathematics?")
chain.invoke(input="I am writing a book and I live in Los Angeles")
chain.invoke(input="My name is Karel,what is the book I am writing about ?")

# Answering the subsequent prompt uses memory.
result=chain.invoke(input="What did we talked about?")
print(result["response"])

