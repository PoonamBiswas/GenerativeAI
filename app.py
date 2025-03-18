import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
import requests

# Load FAISS index
if not os.path.exists("faiss_index/index.faiss"):
    st.error("⚠️ FAISS index not found! Please run `create_index.py` first.")
    st.stop()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
llm = Ollama(model="llama2")

# Define **Weather Forecast Agent**
# def get_weather(city):
#     """Fetch real-time weather data for a city."""
#     api_key = "YOUR_WEATHER_API_KEY"  # Use a valid API Key
#     url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
#     response = requests.get(url)
#     data = response.json()
#     return f"🌤 Weather in {city}: {data['current']['condition']['text']}, {data['current']['temp_c']}°C"

# Define **Fertilizer Expert Agent**
def recommend_fertilizer(crop):
    """Suggest best fertilizer based on the crop."""
    recommendations = {
        "wheat": "Use **NPK 20:20:20** for better yield.",
        "rice": "Use **Urea and DAP** before sowing.",
        "corn": "Apply **Potassium-Rich Fertilizer**.",
    }
    return recommendations.get(crop.lower(), "No specific fertilizer data available.")

# Define **Crop Management Agent**
def crop_advice(crop):
    """Provide farming best practices for different crops."""
    best_practices = {
        "wheat": "🌾 Plant in cool climate. Irrigate twice a week.",
        "rice": "💦 Requires high water levels. Use good quality seeds.",
        "corn": "🌞 Needs full sun and warm temperatures.",
    }
    return best_practices.get(crop.lower(), "No specific crop advice available.")

# Define **Market Dealer Agent**
def get_crop_prices(crop):
    """Fetch current market price of a crop."""
    market_prices = {
        "wheat": "$200 per ton",
        "rice": "$250 per ton",
        "corn": "$180 per ton",
    }
    return f"💰 Current price of {crop}: {market_prices.get(crop.lower(), 'Price not available')}"

# Create Tools for Agents
tools = [
    # Tool(name="Weather Forecast", func=get_weather, description="Get weather updates"),
    Tool(name="Fertilizer Expert", func=recommend_fertilizer, description="Get best fertilizer advice"),
    Tool(name="Crop Manager", func=crop_advice, description="Get crop farming advice"),
    Tool(name="Market Dealer", func=get_crop_prices, description="Get crop market prices"),
]

# Initialize Multi-Agent System
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate.from_template("""
You are an AI assistant helping farmers with agriculture-related queries.
You have access to experts like Fertilizer Expert, Crop Manager, and Market Dealer.

Please answer the question clearly and concisely.

Format:
- **Response:** <Your Answer>
- **Source:** <If applicable>

Question: {query}
""")

agent_executor = initialize_agent(
    tools=tools, 
    llm=llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True, 
    handle_parsing_errors=True, 
    agent_kwargs={"prompt": custom_prompt}  # Use the custom structured prompt
)


# Streamlit UI
st.title("🌾 Agriculture RAG Chatbot with AI Agents")
st.write("Ask any agriculture-related question!")

# User Input
query = st.text_input("Type your question:")

if query:
    response = agent_executor.run(query)
    st.write("🤖 Chatbot Response:")
    st.write(response)
