from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
import os

load_dotenv() # Load environment variables

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# model=ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
model=ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.7)

prompt1 =  ChatPromptTemplate.from_template(
    "Tell me a {adjective} joke about {topic}:"
)

uppercase_step = RunnableLambda(
    lambda x: x.upper()
)

upperlower_step = RunnableLambda(lambda x:lowerupper(x))
uppercase_step = RunnableLambda(lambda x: x.upper())
def route(info):
    if "program" in info:
        return upperlower_step
    else:
        return uppercase_step
def lowerupper(test_str):
    res = ""
    for idx in range(len(test_str)):
        if idx % 2:
            res = res + test_str[idx].upper()
        else:
            res = res + test_str[idx].lower()
    return res

chain =prompt1 | model | StrOutputParser() | RunnableLambda(route)

result = chain.invoke({
    "adjective": "funny",
    "topic": "programming"
})

print(result)

