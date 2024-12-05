import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


def format_file_content(content):
    """
    This function sends the input content to OpenAI for formatting.

    Parameters:
        content (str): The data you want to format as a string.

    Returns:
        str: The formatted content from OpenAI's API response.
    """
    try:
        # Make a request to OpenAI's API
        response = openai.Completion.create(
            engine='text-davinci-003',  # Specify the model/engine to use
            prompt=f"Format the following data consistently: {content}",
            max_tokens=1500  # Limit the response length
        )
        # Extract and return the formatted text from the API response
        return response.choices[0].text.strip()
    except Exception as e:
        # Handle any errors and return an error message
        return f"An error occurred: {str(e)}"
