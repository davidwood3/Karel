import os
from langchain_community.agent_toolkits.load_tools import load_tools
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper

# Load environment variables
load_dotenv()
api_key = os.getenv('SERPER_API_KEY')

tools = load_tools(["google-serper"])

# Example usage with Tool
query = "What is the hometown of the reigning men's U.S. Open champion?"

# Print the response
print(tools[0].invoke(query))
