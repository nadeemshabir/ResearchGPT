"""
LLM Client Module
Unified interface for different LLM APIs (Groq, OpenAI, etc.)
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Unified LLM client supporting multiple providers"""
    
    def __init__(
        self,
        provider: str = "groq",
        model: str = None,
        api_key: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Initialize LLM client
        
        Args:
            provider: LLM provider ('groq', 'openai', 'gemini')
            model: Model name (provider-specific)
            api_key: API key (reads from .env if None)
            temperature: Sampling temperature (0-1)
            max_tokens: Max tokens in response
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        print(f"🤖 Initializing LLM Client...")
        print(f"   Provider: {self.provider}")
        
        # Set default models
        default_models = {
            'groq': 'llama-3.3-70b-versatile',
            'openai': 'gpt-4o-mini',
            'gemini': 'gemini-1.5-flash'
        }
        
        self.model = model or default_models.get(self.provider)
        print(f"   Model: {self.model}")
        
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            key_names = {
                'groq': 'GROQ_API_KEY',
                'openai': 'OPENAI_API_KEY',
                'gemini': 'GOOGLE_API_KEY'
            }
            self.api_key = os.getenv(key_names.get(self.provider))
        
        if not self.api_key:
            raise ValueError(f"No API key found for {self.provider}. Set {key_names.get(self.provider)} in .env")
        
        # Initialize provider-specific client
        self._init_client()
        
        print(f"✅ LLM Client ready")
    
    def _init_client(self):
        """Initialize provider-specific client"""
        if self.provider == 'groq':
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                print("⚠️ Installing groq...")
                os.system("pip install groq")
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
        
        elif self.provider == 'openai':
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("⚠️ Installing openai...")
                os.system("pip install openai")
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
        
        elif self.provider == 'gemini':
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            except ImportError:
                print("⚠️ Installing google-generativeai...")
                os.system("pip install google-generativeai")
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: User prompt
            system_prompt: System instructions, instruction for AI's behavior
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.provider == 'groq':
            return self._generate_groq(prompt, system_prompt, temperature, max_tokens)
        elif self.provider == 'openai':
            return self._generate_openai(prompt, system_prompt, temperature, max_tokens)
        elif self.provider == 'gemini':
            return self._generate_gemini(prompt, system_prompt, temperature, max_tokens)
    
    def _generate_groq(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Groq"""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        """here this chat does not refer to the function created for multi turn chat
        it is the chat function of the LLM client
        """
        
        return response.choices[0].message.content
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using OpenAI"""
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Gemini"""
        # Gemini combines system and user prompts
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.client.generate_content(
            full_prompt,
            generation_config={
                'temperature': temperature,
                'max_output_tokens': max_tokens
            }
        )
        
        return response.text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Multi-turn chat
        
        Args:
            messages: List of {'role': 'user/assistant', 'content': '...'}
            temperature: Override default
            max_tokens: Override default
            
        Returns:
            Assistant response
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        if self.provider in ['groq', 'openai']:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == 'gemini':
            # Convert to Gemini format
            chat = self.client.start_chat(history=[])
            for msg in messages[:-1]:  # All but last
                if msg['role'] == 'user':
                    chat.send_message(msg['content'])
            
            # Send last message and get response
            response = chat.send_message(messages[-1]['content'])
            return response.text
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'provider': self.provider,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }


def test_llm_client():
    """Test LLM client"""
    
    print("\n" + "="*80)
    print("TESTING LLM CLIENT")
    print("="*80)
    
    # Test with Groq (free)
    try:
        client = LLMClient(provider='groq')
        
        # Simple generation
        print("\n1️⃣ Testing simple generation...")
        response = client.generate(
            prompt="What is attention mechanism in transformers? Answer in 2 sentences.",
            system_prompt="You are a helpful AI research assistant."
        )
        print(f"\nResponse:\n{response}")
        
        # Multi-turn chat
        print("\n2️⃣ Testing multi-turn chat...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is BERT?"},
        ]
        response = client.chat(messages)
        print(f"\nResponse:\n{response}")
        
        # Token counting
        print("\n3️⃣ Testing token counting...")
        text = "This is a sample text for token counting."
        tokens = client.count_tokens(text)
        print(f"Text: {text}")
        print(f"Estimated tokens: {tokens}")
        
        # Model info
        print("\n4️⃣ Model info:")
        info = client.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure you have set GROQ_API_KEY in your .env file")
        print("Get free API key from: https://console.groq.com/")


if __name__ == "__main__":
    test_llm_client()