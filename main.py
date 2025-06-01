
import os
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# Your imports from the agent framework
from agents import Agent, RunConfig, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

# Load environment variables
load_dotenv(find_dotenv())

# Get API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please define it in your .env file.")

# Initialize FastAPI app
app = FastAPI()

# Allow frontend (e.g., Next.js) to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use a specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Agent system
provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

agent = Agent(
    name="Food Analyzer",
    instructions="You are a helpful assistant that analyzes food items and provides nutritional information. Please answer the user's questions only about food items accurately and concisely."
)

# API input format
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]  # [{"role": "user", "content": "What's in an apple?"}]

# Endpoint to receive chat messages and respond
@app.post("/chat")
async def chat(req: ChatRequest):
    result = await Runner.run(agent, input=req.messages, run_config=config)
    return {"response": result.final_output}
