import os
import pandas as pd
from django.shortcuts import render
from dotenv import load_dotenv
from .forms import FileUploadForm
from .models import UploadedFile
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=api_key)

def upload_file(request):
    """
    Handle file uploads and process them using OpenAI API.
    """
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded file
            uploaded_file = form.save()
            file_path = uploaded_file.file.path

            # Process the uploaded file
            analysis_result = process_file(file_path)

            # Render the result page with the analysis
            return render(request, 'analysis_result.html', {
                'analysis_result': analysis_result,
            })
    else:
        form = FileUploadForm()
    return render(request, 'upload.html', {'form': form})

def process_file(file_path):
    """
    Read the uploaded file and perform analysis using OpenAI.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        logging.debug(f"DataFrame loaded: {df.head()}")

        # Convert the DataFrame to a CSV string
        csv_data = df.to_csv(index=False)

        # Generate analysis using OpenAI API
        return analyze_data_with_openai(csv_data)
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return f"An error occurred while processing the file: {str(e)}"

def analyze_data_with_openai(data):
    """
    Use OpenAI to analyze the given data and provide insights.
    """
    prompt = (
        "You are a data analyst. You are analysing student grades in IB Diploma Programme."
        "We are looking for a detailed numbered analysis of anything you find notable in this document. Analyze the following data and provide insights, "        
        "including trends, key metrics, and notable observations."
        "You can even try to compare data to links in the same document, or draw conclusions why students might be doing good or bad, as well as possible solutions to that.:\n\n"
        f"{data}\n\n"
        "Highlight key points and provide a concise summary"
    )

    try:
        # Call OpenAI's API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return f"An error occurred while analyzing the data: {str(e)}"