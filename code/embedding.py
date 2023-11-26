from typing import List, Union
#from InstructorEmbedding import INSTRUCTOR
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import json
from translation import NLPTranslation
import aiofiles
import asyncio
import time

async def generate_embed_openai(text, openai_api_key, language, model= "text-embedding-ada-002", retry_times=3) -> Union[List, str]:
    openai.api_key = openai_api_key
    
    # Check if the input text is None or empty, raise an error if so
    if text is None or text.strip() == '':
        raise ValueError("Input text cannot be None or empty.")

    for i in range(retry_times):
        try:
            # Replace newline characters with spaces for consistent formatting
            text = text.replace("\n", " ")
            # Generate and return the embedding using OpenAI API
            embed = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
            result = {"embed": embed}
            return result
        except Exception as e:
            # If any error occurs during embedding generation, log the error and retry
            if i < retry_times - 1:  # If it's not the last attempt
                await asyncio.sleep(1)  # Wait for 1 second before retrying
            else:  # If it's the last attempt and still failing, raise an error
                raise RuntimeError(f"An error occurred while generating the embedding after {retry_times} attempts: {str(e)}")

def compare_embeds(embed1: np.ndarray, embed2: np.ndarray) -> Union[float, str]:
    # Check if the embeddings are empty, raise an error if so
    if len(embed1) == 0 or len(embed2) == 0:
        raise ValueError("Embeddings cannot be empty.")
    # Check if the embeddings are of the same length, raise an error if not
    if len(embed1) != len(embed2):
        raise ValueError("Embeddings must be of the same length.")
    try:
        # Calculate the cosine similarity between the two embeddings
        similarity = cosine_similarity(embed1, embed2)
        # Ensure the similarity is within the range [0,1]
        similarity = np.clip(similarity, 0, 1)
        # Return the cosine similarity
        return similarity[0][0]
    except Exception as e:
        # If any error occurs during cosine similarity calculation, raise a runtime error
        raise RuntimeError(f"An error occurred while calculating cosine similarity: {str(e)}")

async def language_embed(language, file_name = "language_schema.json") -> bool:
    try:
        async with aiofiles.open(file_name, mode='r') as f:
            schema = await f.read()
        schema = json.loads(schema)
        if language not in schema:
            raise ValueError(f"Language '{language}' not found in schema.")
        elif "embed translate" not in schema[language]:
            raise ValueError(f"Key 'embed_translate' not found for language '{language}' in schema.")
        else:
            return schema[language]["embed translate"]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File '{file_name}' not found.") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON in file '{file_name}': {str(e)}") from e


"""
async def generate_embed(text) -> Union[np.ndarray, str]:
    # Load the Instructor model
    instructor_model = INSTRUCTOR('hkunlp/instructor-xl')
    
    # Check if the input text is None or empty, raise an error if so
    if text is None or text.strip() == '':
        raise ValueError("Input text cannot be None or empty.")
    try:
        # Replace newline characters with spaces for consistent formatting
        text = text.replace("\n", " ")
        # Generate and return the embedding
        embed = instructor_model.encode([["Represent the text:", text]])
        result = {"embed": embed}
        return result
    except Exception as e:
        # If any error occurs during embedding generation, raise a runtime error
        raise RuntimeError(f"An error occurred while generating the embedding: {str(e)}")
"""





