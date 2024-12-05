from langchain.memory import ConversationBufferWindowMemory
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
    verbose=True,
    # memory = ConversationBufferWindowMemory(k=2)
    memory = ConversationBufferWindowMemory()
)

chain.invoke(
input="I am writing a book about mathematics, can you explain what is mathematics?")
chain.invoke(input="I am writing a book and I live in Los Angeles")
chain.invoke(input="My name is Karel,what is the book I am writing about ?")

# Answer the subsequent prompt using memory.
result = chain.invoke(input="What did we talk about?")
# print (result["response"])
print(json.dumps(result, indent=4))


