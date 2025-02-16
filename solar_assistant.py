import gradio as gr
import os
import json
import logging
import httpx
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

class SolarAssistant:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found in environment variables")
        self.api_base = "https://openrouter.ai/api/v1"
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='solar_assistant.log'
        )
    
    def _generate_system_prompt(self):
        return """You are a specialized solar industry consultant AI assistant. Provide accurate, practical advice while adapting your responses to the user's technical expertise level."""
    
    async def _async_get_response(self, query, user_expertise="general"):
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost:7860"
            }
            
            # Using a more economical model
            data = {
                "model": "mistralai/mistral-7b-instruct",  # More economical model
                "messages": [
                    {"role": "system", "content": self._generate_system_prompt()},
                    {"role": "user", "content": f"User expertise level: {user_expertise}\nQuery: {query}"}
                ],
                "temperature": 0.7,
                "max_tokens": 500  # Reduced token limit
            }
            
            logging.debug(f"Sending request with data: {json.dumps(data, indent=2)}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                logging.debug(f"Response status code: {response.status_code}")
                response_text = response.text
                logging.debug(f"Raw response: {response_text}")
                
                response_data = response.json()
                
                if 'error' in response_data:
                    error_msg = response_data['error'].get('message', 'Unknown error')
                    return f"API error: {error_msg}"
                
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    choice = response_data['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        return choice['message']['content']
                    elif 'text' in choice:
                        return choice['text']
                
                return "Error: Unexpected response format from API"
                
        except Exception as e:
            logging.exception("Error in API call:")
            return f"An error occurred: {str(e)}"

    def get_response(self, query, user_expertise="general"):
        """Synchronous wrapper for async API call"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self._async_get_response(query, user_expertise))
            return response
        finally:
            loop.close()

class SolarAssistantUI:
    def __init__(self):
        self.assistant = SolarAssistant()
    
    def create_interface(self):
        return gr.Interface(
            fn=self.assistant.get_response,
            inputs=[
                gr.Textbox(label="Ask your solar-related question", placeholder="Type your question here..."),
                gr.Radio(
                    choices=["general", "technical", "expert"],
                    label="Your expertise level",
                    value="general"
                )
            ],
            outputs=gr.Textbox(label="Response"),
            title="Solar Industry AI Assistant",
            description="Get expert advice on solar technology, installation, maintenance, and more.",
            examples=[
                ["What factors should I consider when installing solar panels?", "general"],
                ["Calculate the ROI for a 10kW solar system in California", "technical"],
                ["Latest efficiency improvements in PERC solar cells?", "expert"]
            ],
            theme=gr.themes.Base()
        )

def main():
    print("Starting Solar Industry AI Assistant...")
    ui = SolarAssistantUI()
    interface = ui.create_interface()
    print("Launching interface...")
    interface.launch(debug=True, show_error=True)
    print("Interface launched!")

if __name__ == "__main__":
    main()