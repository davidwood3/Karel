import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain_core.messages import (HumanMessage, AIMessage)
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.runnables import RunnableLambda

def setup_environment():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)
    return api_key

def initialize_document_store(documents_path):
    # Ensure documents directory exists
    documents_dir = Path(documents_path)
    if not documents_dir.exists():
        print(f"Creating documents directory at {documents_dir}")
        documents_dir.mkdir(parents=True, exist_ok=True)
        
    try:
        # First, manually verify each file
        documents = []
        for txt_file in documents_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(Document(page_content=content, metadata={"source": str(txt_file)}))
                print(f"Successfully loaded {txt_file.name}")
            except Exception as e:
                print(f"Error loading {txt_file.name}: {str(e)}")
                continue
        
        print(f"Loaded {len(documents)} documents from {documents_dir}")
        if not documents:
            print("Warning: No documents found in the documents directory")
            print(f"Please add .txt files to: {documents_dir}")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def setup_vector_store(documents, persist_dir):
    # Create persist directory if it doesn't exist
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    # Initialize text splitter and embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200  # Reduced overlap to save memory
    )
    embeddings = OpenAIEmbeddings()

    # Process documents if any exist
    if documents:
        chunks = text_splitter.split_documents(documents)
        return Chroma.from_documents(
            documents=chunks,
            persist_directory=str(persist_path),
            embedding=embeddings,
            collection_name="conversations"
        )
    else:
        # Initialize empty vector store
        return Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings,
            collection_name="conversations"
        )

def chroma_retriever(input):
    query = input["input"]
    try:
        results = vector_store.search(query=query, search_type="mmr", k=5)  # Reduced k for better performance
        documents = [result.page_content for result in results]
        return {"documents": documents, "input": input["input"], "history": input["history"]}
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return {"documents": [], "input": input["input"], "history": input["history"]}

def savechattochroma(human, ai):
    try:
        docsx = []
        human_content = human.content if isinstance(human, HumanMessage) else str(human)
        ai_content = ai.content if isinstance(ai, AIMessage) else str(ai)
        
        docsx.append(Document(page_content=f"human: {human_content}"))
        docsx.append(Document(page_content=f"AI: {ai_content}"))
        vector_store.add_documents(documents=docsx)
        # vector_store.persist()  # Ensure changes are saved
    except Exception as e:
        print(f"Error saving chat history: {e}")

def process_chat(runnable, user_input):
    try:
        response = runnable.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        ).content
        return response
    except Exception as e:
        print(f"Error processing chat: {e}")
        return "I apologize, but I encountered an error processing your request."

def main():
    print("Initializing chat system...")
    
    # Setup paths
    base_dir = Path(__file__).parent
    documents_path = base_dir.parent / "Ch08" / "documents"
    persist_dir = base_dir / "chroma_db"
    
    # Initialize components
    api_key = setup_environment()
    documents = initialize_document_store(documents_path)
    
    global vector_store
    vector_store = setup_vector_store(documents, persist_dir)
    
    # Initialize LLM and prompt
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)  # Using 3.5-turbo for cost efficiency
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Your responses are based on the 
        following context from previous conversations: {documents}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Setup chat components
    retriever = RunnableLambda(chroma_retriever)
    memory = ChatMessageHistory()
    chain = retriever | prompt | llm
    
    global session_id
    session_id = "user_123"
    
    runnable = RunnableWithMessageHistory(
        chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="history"
    )

    print("\nAgent is ready to chat. Type 'exit' or 'quit' to end the conversation.")
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                continue
                
            response = process_chat(runnable, user_input)
            print(f"\nAI: {response}")
            
            savechattochroma(
                HumanMessage(content=user_input),
                AIMessage(content=response)
            )
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    finally:
        vector_store.persist()  # Ensure final save
        print("Chat session ended.")

if __name__ == "__main__":
    main()
