from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from Ch05.PromptTemp2 import input_variables

load_dotenv() # Load environment variables

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

template = """# Define a more complex prompt template
Role: {role}
User: {user}
Task: {task}
"""

prompt_template = ChatPromptTemplate.from_template(template)
# Create a chat model instance
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")

input_variables = {# Provide a list of input variables}
    "role": "Teacher",
    "user": "Alice",
    "task": "Explain the theory of relativity."
}

# Generate the final prompt
formatted_prompt = prompt_template.invoke(input=input_variables)

# Send the message to the model and get the response
response = llm.invoke(formatted_prompt)
print(response.content)
