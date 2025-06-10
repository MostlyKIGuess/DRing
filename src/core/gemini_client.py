"""
Core Gemini API client for Dread Rising
"""
import streamlit as st
from google import genai
from google.genai import types
from typing import Optional, List, Dict, Any
from config.settings import GEMINI_MODEL, GEMINI_TEMPERATURE, MAX_OUTPUT_TOKENS


class GeminiClient:
    """Centralized Gemini API client"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client"""
        try:
            self.client = genai.Client(api_key=self.api_key)
            return True
        except Exception as e:
            st.error(f"Error initializing Gemini client: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """Check if client is properly initialized"""
        return self.client is not None
    
    def generate_content(
        self, 
        contents: List[Any], 
        system_instruction: Optional[str] = None,
        temperature: float = GEMINI_TEMPERATURE,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        tools: Optional[List[types.Tool]] = None
    ) -> Optional[str]:
        """Generate content using Gemini"""
        if not self.client:
            return None
        
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            if system_instruction:
                config.system_instruction = system_instruction
                
            if tools:
                config.tools = tools
            
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config
            )
            
            return response.text
        except Exception as e:
            st.error(f"Error generating content: {str(e)}")
            return None
    
    def generate_with_search(
        self, 
        query: str, 
        system_instruction: Optional[str] = None
    ) -> Optional[Dict]:
        """Generate content with Google Search grounding"""
        if not self.client:
            return None
        
        try:
            google_search_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            config = types.GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )
            
            if system_instruction:
                config.system_instruction = system_instruction
            
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[query],
                config=config
            )
            
            result = {
                "text": response.text,
                "grounding_metadata": None
            }
            
            #grounding metadata if available
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    result["grounding_metadata"] = candidate.grounding_metadata
            
            return result
        except Exception as e:
            st.error(f"Error generating grounded content: {str(e)}")
            return None
    
    def generate_with_url_context(
        self, 
        query: str, 
        urls: List[str] = None,
        use_search: bool = False
    ) -> Optional[Dict]:
        """Generate content with URL context"""
        if not self.client:
            return None
        
        try:
            tools = [types.Tool(url_context=types.UrlContext)]
            
            if use_search:
                tools.append(types.Tool(google_search=types.GoogleSearch))
            
            config = types.GenerateContentConfig(
                tools=tools,
                response_modalities=["TEXT"],
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )
            
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",  # URL context requires this model more info on settings.py
                contents=[query],
                config=config
            )
            
            result = {
                "text": response.text,
                "url_metadata": None
            }
            
            # URL metadata if available
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'url_context_metadata'):
                    result["url_metadata"] = candidate.url_context_metadata
            
            return result
        except Exception as e:
            st.error(f"Error generating content with URL context: {str(e)}")
            return None
    
    def analyze_pdf(
        self, 
        pdf_bytes: bytes, 
        prompt: str,
        system_instruction: Optional[str] = None
    ) -> Optional[str]:
        """Analyze PDF with Gemini"""
        if not self.client:
            return None
        
        try:
            contents = [
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type='application/pdf',
                ),
                prompt
            ]
            
            return self.generate_content(
                contents=contents,
                system_instruction=system_instruction
            )
        except Exception as e:
            st.error(f"Error analyzing PDF: {str(e)}")
            return None
