import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from split_text import split_text_by_chars
import asyncio
from embedding import generate_embed_openai, language_embed
from translation import NLPTranslation
import urllib.parse

# MAIN:
async def Embed_website(file_name, NLP_api_key, openai_api_key, language):
    # Dictionary mapping domain names to their respective functions
    domain_to_functions = {
        'example.com': embed_example,
        # Add more domains and their respective functions here
    }

    _, domain = os.path.splitext(file_name)

    try:
        if is_valid_website(file_name):
            if domain in domain_to_functions:
                # If the domain exists, call the respective function
                output = await domain_to_functions[domain](file_name, openai_api_key, NLP_api_key, language)
                return output
            else:
                # Call the generic function for other domains
                output = await embed_regular_website(file_name, openai_api_key, NLP_api_key, language)
                return output
        else:
            # raise Exception("The provided link is not a valid website.")
            return None  # Return None if the link is not a valid website
    except Exception as e:
        # Raise a regular exception without the custom message
        raise Exception("An error occurred while embedding the website.") from e

def is_valid_website(link):
    parsed_url = urllib.parse.urlparse(link)
    return parsed_url.scheme in ('http', 'https') and parsed_url.netloc
    
# Example function
async def embed_example(file_name, openai_api_key, NLP_api_key, language):
    return None

#######################################################################################################################################
################################################### For dictionary Structure ##########################################################
#######################################################################################################################################

async def create_text_content(text, english_text, embed):
    return {
        'class': "text",
        'text': text,
        'english text': english_text, # Only exist if embed/ai language is in English
        'embed': embed
    }

async def create_result(url, description, language, embed_language, content):
    return {
        'class': "website",
        'description': description,
        'url': url,
        'language': {'prefix:': language, 'ai language': "",'embed language': embed_language},
        'content': content
    }

#######################################################################################################################################
################################################### For extracting website ############################################################
#######################################################################################################################################

# NEED UPDATING WONT BE IN USE
async def download_images(url, soup):
    # Find all images in the parsed HTML
    images = soup.find_all('img')
    for image in images:
        # Get the src attribute of the image
        img_url = image.get('src')
        # Create a full url by joining the base url with the img_url
        full_url = urljoin(url, img_url)
        # Get the file name
        file_name = os.path.basename(full_url)
        # Download the image
        img_data = requests.get(full_url).content
        if file_name.strip():
            with open(file_name, 'wb') as handler:
                handler.write(img_data)

async def embed_regular_website(url, NLP_api_key, openai_api_key, language):
    # Send a GET request to the website
    response = requests.get(url)
    # Parse the HTML content of the website
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract and print text content
    text = soup.get_text()

    # Download images (NEED UPDATING)
    # download_images(url, soup)
    
    embed_translate = await language_embed(language)
    if embed_translate:
        embed_language = 'en'
    else: 
        embed_language = language
    content = {}
    content_id = 0  # Initialize counter for unique content IDs

    page_content = await extract_page_content(text, NLP_api_key, openai_api_key, language, embed_translate)

    for content_piece in page_content:
        content[content_id] = content_piece
        content_id += 1

    description = ""
    document = await create_result(url, description, language, embed_language, content)
    
    return document

# file_path: str, openai_api_key: str, NLP_api_key: str, language: str
async def extract_page_content(text, NLP_api_key, openai_api_key, language, embed_translate):
    # Text is fed on a page by page basis, page_num will equal -1 if the text has no page structure. 
    # Otherwise it starts from 0 for page 1.
    english_text = ""

    # If embed_translate flag is True, text is translated to English.
    if embed_translate:
        try:
            # Split text into pieces of up to 1000 characters each.
            texts_list = await split_text_by_chars(text, 1000)
            
            # Translate each piece to English using the NLP API.
            english_text_list = [await NLPTranslation(text, NLP_api_key, language, "en") for text in texts_list]
            
            # Extract the translated text from each response.
            english_text_list = [response["translated text"] for response in english_text_list]
        except Exception as e:
            raise RuntimeError(f"An error occurred while translating the text: {str(e)}")
    else:
        # If not translating, just split text into pieces of up to 1000 characters each.
        texts_list = await split_text_by_chars(text, 1000)

    # Prepare to store content for each piece of text.
    page_content = []

    # Decide whether to use translated or original text for next steps.
    if embed_translate:
        texts = english_text_list
    else:
        texts = texts_list
        
    # Generate an embedding for each piece of text and create content for it.
    for i, text in enumerate(texts):
        # Use OpenAI API to generate embedding for the text.
        embed_result = await generate_embed_openai(text, openai_api_key, language)

        # If embed_translate flag is True, include translated text in the content.
        if embed_translate:
            english_text = text
            text = texts_list[i]
        else:
            english_text = ""

        # Create content for this piece of text including the embedding and possibly the translation.
        content_piece = await create_text_content(text, english_text, embed_result["embed"])

        # Add the content piece to the list.
        page_content.append(content_piece)

    # Return the list of content pieces for this page.
    return page_content

"""

import json

async def call():
    openai_api_key = "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"
    NLP_api_key = "2d786c36fcmshf548a3d9bca8368p1e4c15jsnca35ccbad050"
    language = "th"
    url = "https://www.scimath.org/lesson-mathematics/item/7334-2017-06-17-14-37-32"
    output_dict = await Embed_website(url, NLP_api_key, openai_api_key, language)
        
    # Define the output file path
    output_file = "output.json"
    
    # Write the dictionary to a JSON file
    with open(output_file, "w") as f:
        json.dump(output_dict, f)
    
asyncio.run(call())
"""





