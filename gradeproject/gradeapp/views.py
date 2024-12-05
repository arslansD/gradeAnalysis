import os
import pandas as pd
from django.shortcuts import render
from dotenv import load_dotenv
from .forms import FileUploadForm
from .models import UploadedFile
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

# Initialize the OpenAI client
openai.api_key = api_key

def upload_file(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            file_path = uploaded_file.file.path
            # Process the file
            analysis_result, images = process_file(file_path)
            return render(request, 'analysis_result.html', {
                'analysis_result': analysis_result,
                'images': images
            })
    else:
        form = FileUploadForm()
    return render(request, 'upload.html', {'form': form})


def process_file(file_path):
    # Read the file into a DataFrame
    try:
        df = pd.read_csv(file_path)
    except pd.errors.ParserError:
        return "Error: The file could not be parsed. Please check the file format.", []
    except FileNotFoundError:
        return "Error: The file was not found. Please check the file path.", []

    # Generate textual analysis using GPT-4
    analysis_result = generate_textual_analysis(df)

    # Generate visualizations
    images = generate_visualizations(df)

    return analysis_result, images


def generate_textual_analysis(df):
    # Convert DataFrame to CSV string
    csv_data = df.to_csv(index=False)

    # Create a prompt for OpenAI
    prompt = (
        "You are a data analyst. Analyze the following student results data and provide insights, "
        "including trends, comparisons between different years, and any notable observations. "
        "Present the analysis in a structured format:\n\n"
        f"{csv_data}\n\n"
        "Provide the analysis below:"
    )

    # Call OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.5
        )
        # Extract and return the analysis
        analysis = response.choices[0].message['content'].strip()
    except Exception as e:
        analysis = f"An error occurred while processing the file: {str(e)}"

    return analysis


def generate_visualizations(df):
    images = []

    # Ensure there are at least two columns to plot
    if df.shape[1] < 2:
        print("Not enough columns to generate a scatter plot.")
        return images

    # Select the first two columns for the scatter plot
    x_col = df.columns[0]
    y_col = df.columns[1]

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f'Scatter Plot of {x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode the image to base64
    image_base64 = base64.b64encode(image_png).decode('utf-8')
    images.append(image_base64)

    # Close the plot to free memory
    plt.close()

    return images