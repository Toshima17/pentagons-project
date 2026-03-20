import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 1. Setup API Key (This pulls from Hugging Face 'Secrets')
# Make sure to add GROQ_API_KEY in your Space Settings!
groq_api_key = os.getenv("GROQ_API_KEY")

# 2. Initialize the LLM (Using the Llama3 model from your notebook)
llm = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="llama3-8b-8192",
    temperature=0.5
)

# 3. Setup Embeddings (Same as your notebook)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Define the Chat Logic
# This function handles the conversation history and the AI response
def chat_response(message, history):
    try:
        # Convert Gradio history format to LangChain format if needed
        # For a simple version, we'll invoke the LLM directly:
        response = llm.invoke(message)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}. Please check if your GROQ_API_KEY is set in Space Secrets."

# 5. Create the Gradio Web Interface
demo = gr.ChatInterface(
    fn=chat_response,
    title="Mental Check-In Journal Bot",
    description="I am your AI companion for mental well-being. How are you feeling today?",
    theme="soft"
)

# 6. Launch the App
if __name__ == "__main__":
    demo.launch()
