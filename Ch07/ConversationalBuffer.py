from langchain.memory import ConversationBufferMemory
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
    memory = ConversationBufferMemory()
)

chain.invoke("The book topic is mathematics, can you explain what is mathematics?")

# Answer the subsequent prompt using memory.
result = chain.invoke(input="What is the book about?")
# print (result["response"])
print(json.dumps(result, indent=4))


