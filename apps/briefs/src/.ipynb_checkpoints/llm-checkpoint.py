import google.generativeai as genai
from google.api_core import exceptions as google_exceptions # Import specific exceptions
import os
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Tuple, Optional
import sys

# --- Load .env file ---
# Find the .env file starting from this script's directory and going up
dotenv_path = find_dotenv()
if dotenv_path:
    print(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path, override=True) # Load the found file
else:
    print("Warning: .env file not found by find_dotenv(). Trying default load.")
    try:
        load_dotenv(override=True)
        print("Attempted default load_dotenv().")
    except Exception as e:
        print(f"Default load_dotenv() also failed: {e}")

# --- Configure Google API Key ---
# This configures the library globally based on the loaded environment variable
IS_API_KEY_CONFIGURED = False
try:
    api_key = os.environ["GOOGLE_API_KEY"] # Check if key exists in environment
    genai.configure(api_key=api_key)
    print("Google API Key configured successfully.")
    IS_API_KEY_CONFIGURED = True # Set flag if successful
except KeyError:
    print("ERROR: GOOGLE_API_KEY environment variable not found.")
    # Consider raising an error if the key is absolutely essential for the app to function
    # raise EnvironmentError("GOOGLE_API_KEY required but not found.")
except Exception as e:
    print(f"Error configuring Google API Key during initial setup: {e}")

# --- Reusable function to call the Gemini model ---
def call_llm(model: str, messages: List[Dict], temperature: float = 0.0) -> Tuple[str, Optional[dict]]:
    """
    Calls the specified Google Gemini model.

    Args:
        model: The name of the model to use (e.g., 'gemini-1.5-flash').
        messages: A list of message dictionaries, EXPECTED IN OPENAI FORMAT
                  (e.g., {'role': 'user'/'assistant', 'content': 'text'}).
                  This function handles conversion to Google's format.
        temperature: The generation temperature (clamped between 0.0 and 1.0).

    Returns:
        A tuple containing:
        - The generated text content as a string (or an error message string).
        - Usage metadata dictionary (or None if error or not available).
    """
    print(f"Attempting to call model: {model} with temperature: {temperature}")

    # Check if initial configuration succeeded
    if not IS_API_KEY_CONFIGURED:
         print("ERROR: Cannot call LLM, Google API Key was not configured successfully.")
         return "Error: Library not configured", None

    google_contents = [] # Initialize empty list for converted format

    try:
        # --- Convert OpenAI message format to Google 'contents' format ---
        for msg in messages:
            if 'role' in msg and 'content' in msg:
                role = 'model' if msg['role'] == 'assistant' else msg['role'] # Map role
                # Ensure content is treated as simple text for the 'parts' list
                google_contents.append({'role': role, 'parts': [{'text': str(msg['content'])}]})
            else:
                print(f"Warning: Skipping message with invalid format: {msg}")
        # --- End Conversion ---

        if not google_contents:
             print("Error: No valid messages found after format conversion.")
             return "Error: No valid messages to send", None

        # Clamp temperature
        temperature = max(0.0, min(1.0, temperature))

        gemini_model = genai.GenerativeModel(model_name=model)

        print(f"Sending converted contents format: {google_contents}") # Log the converted format

        response = gemini_model.generate_content(
            contents=google_contents, # Pass the converted list
            generation_config=genai.types.GenerationConfig(
                temperature=temperature
                )
            )

        usage_data = getattr(response, 'usage_metadata', None)

        # Safely extract text
        if response.candidates:
            if response.candidates[0].content and response.candidates[0].content.parts:
                generated_text = "".join(part.text for part in response.candidates[0].content.parts)
                print("LLM call successful.")
                return generated_text, usage_data
            else:
                finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
                safety_ratings = response.candidates[0].safety_ratings if response.candidates else 'UNKNOWN'
                print(f"LLM call finished but no content parts found. Finish Reason: {finish_reason}, Safety Ratings: {safety_ratings}")
                return f"Error: Generation finished without content. Reason: {finish_reason}", usage_data
        else:
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'UNKNOWN'
            safety_ratings = response.prompt_feedback.safety_ratings if response.prompt_feedback else 'UNKNOWN'
            print(f"LLM call failed. Prompt Feedback Block Reason: {block_reason}, Safety Ratings: {safety_ratings}")
            return f"Error: LLM call failed. Prompt Block Reason: {block_reason}", usage_data

    # --- Specific Error Handling for API Calls ---
    except google_exceptions.PermissionDenied as e:
        print(f"ERROR: Gemini API Permission Denied (Check API Key/Project Billing?): {e}")
        return f"Error: Permission Denied - {e}", None
    except google_exceptions.ResourceExhausted as e:
        print(f"ERROR: Gemini API Quota Exceeded: {e}")
        return f"Error: Quota Exceeded - {e}", None
    except google_exceptions.NotFound as e:
         print(f"ERROR: Gemini API resource not found (Check model name '{model}' or endpoint): {e}")
         return f"Error: Not Found - {e}", None
    except google_exceptions.InvalidArgument as e:
         print(f"ERROR: Invalid argument sent to Gemini API (Check messages format?): {e}")
         print(f"Problematic Original messages variable: {messages}")
         print(f"Problematic Converted contents variable: {google_contents}")
         return f"Error: Invalid Argument - {e}", None
    except Exception as e: # Generic catch-all last
        # Use _class.name_ which exists on all exception types
        print(f"ERROR: Unexpected error during Gemini API call: {e._class.name_} - {e}")
        print(f"Original messages at time of error: {messages}")
        print(f"Converted contents at time of error: {google_contents}")
        return f"Error calling LLM: {e._class.name_} - {e}", None

# --- Remove any old Pydantic models if they are no longer needed ---
# class Story(BaseModel):
#    title: str = Field(description="title of the story")