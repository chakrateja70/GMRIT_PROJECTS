# services/gemini_llm_service.py

from google import genai
from google.genai import types
from src.core.exceptions import LLMServiceAPIException, LLMServiceUnexpectedException
from src.core.prompts import ANSWER_PROMPT_TEMPLATE
from src.config.settings import GEMINI_API_KEY, GEMINI_LLM_MODEL 

class GeminiLLMService:
    """
    Service for interacting with the Gemini LLM (gemini-2.5-flash).
    Implements a Singleton pattern to ensure only one client is initialized.
    """

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiLLMService, cls).__new__(cls)
            cls._client = genai.Client(api_key=GEMINI_API_KEY)
        return cls._instance

    def generate_answer(self, context: str, question: str) -> str:
        """
        Generates an answer using Gemini 2.5 Flash model.
        """
        if self._client is None:
            raise LLMServiceUnexpectedException("Gemini client is not initialized")
        
        try:
            formatted_prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=formatted_prompt)],
                ),
            ]

            # Configure generation parameters including max tokens
            generate_content_config = types.GenerateContentConfig(
                max_output_tokens=4000, 
                temperature=0.5,  # Added temperature for consistent responses
                top_p=0.95,  # Added for better quality
            )

            # Stream the response and collect chunks
            response_text = ""
            for chunk in self._client.models.generate_content_stream(
                model=GEMINI_LLM_MODEL,
                contents=contents,
                config=generate_content_config,
            ):
                if hasattr(chunk, "text") and chunk.text:
                    response_text += chunk.text

            return response_text.strip() if response_text else "No response from Gemini model."

        except Exception as e:
            # Differentiate between API and unexpected errors if needed
            if "401" in str(e) or "403" in str(e):
                raise LLMServiceAPIException(f"Gemini API Error: {e}")
            raise LLMServiceUnexpectedException(f"Unexpected Gemini LLM Error: {e}")


# Create a single shared instance
gemini_llm_service = GeminiLLMService()

llm_service = gemini_llm_service