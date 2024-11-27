from langchain_anthropic import ChatAnthropic

#from langchain_community.chat_models.anthropic import ChatAnthropic
# from langchain_anthropic.ChatAnthropic import ChatAnthropic

import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables
# Get OpenAI API key from environment variables
claude_api_key   = os.getenv("CLAUDE_API_KEY2")
print(claude_api_key)

# Create a chat model instance
# model = ChatAnthropic(anthropic_api_key=claude_api_key, model="claude-3-sonnet-20240229")
model = ChatAnthropic(anthropic_api_key=claude_api_key, model="claude-3-5-haiku-20241022")

print("Q and A with AI")
print("================")

# Define the question
question = "Who is Elon Musk?"
print("Question:", question)

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
response = model.invoke(messages)

# Generate the answer
# response = model.invoke(question)
print("Answer:", response.content)
