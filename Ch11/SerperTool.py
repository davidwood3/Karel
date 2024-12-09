import os
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper

# Load environment variables
load_dotenv()
api_key = os.getenv('SERPER_API_KEY')

# Initialize Serper Search tool
search_tool = GoogleSerperAPIWrapper()

# Example usage
query = "What is the hometown of the reigning men's U.S. Open champion?"
response = search_tool.run(query)

# Print search results
print(response)

