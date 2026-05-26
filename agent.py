import os
from dotenv import load_dotenv
from siphon import Agent
from siphon.plugins import deepgram, cartesia, azure_openai

load_dotenv()

# Instantiate your models
llm_model = azure_openai.LLM(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_BASE_URL"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
stt_model = deepgram.STT()
tts_model = cartesia.TTS(voice="f786b574-daa5-4673-aa0c-cbe3e8534c02")

# Create the Agent
agent = Agent(
    agent_name="thruv-dev-agent",
    llm=llm_model,  
    stt=stt_model,
    tts=tts_model,
    system_prompt="You are a helpful and professional enterprise AI receptionist. Keep your answers brief and conversational."
)

if __name__ == "__main__":
    # Download required models/dependencies (Uncomment and run this ONLY for the first-time setup)
    # agent.download()

    # Start the worker node (auto-connects to the Siphon dispatcher)
    agent.start()