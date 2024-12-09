import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()
api_key = os.getenv('TAVILY_API_KEY')

# Initialize Tavily Search tool
search_tool = TavilySearchResults()

# Example usage
query = "What countries won the most gold in 2022?"

# Print search results
print(search_tool.invoke(query))
