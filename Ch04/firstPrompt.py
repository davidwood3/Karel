from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables
# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create a chat model instance
chat_model = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4")

print("Q and A with AI")
print("================")

# Define the question
question = "What is the capital of France?"
print("Question:", question)

# Generate the answer
response = chat_model.invoke(question)
print("Answer:", response.content)