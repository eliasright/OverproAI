import asyncio
import os
import json
from pathlib import Path
from embed_spreadsheet import Embed_spreadsheet
from embed_website import Embed_website
from embed_document import Embed_document
from embed_multimedia import Embed_multimedia
from embedding import generate_embed_openai, compare_embeds
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


async def Embed_any(file_path, openai_api_key, NLP_api_key, deepgram_api_key, language):
    result = await Embed_spreadsheet(file_path, NLP_api_key, openai_api_key, language)
    if result is not None:
        return result
    
    result = await Embed_website(file_path, NLP_api_key, openai_api_key, language)
    if result is not None:
        return result
    
    result = await Embed_document(file_path, NLP_api_key, openai_api_key, language)
    if result is not None:
        return result
    
    result = await Embed_multimedia(file_path, deepgram_api_key, NLP_api_key, openai_api_key, language)
    if result is not None:
        return result
    
    return None

# TESTING AREA
async def main(folder_path, openai_api_key, NLP_api_key, deepgram_api_key, language):
    files = [f for f in Path(folder_path).iterdir() if f.is_file()]
    output_folder = 'memory'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        counter = 1
        while os.path.exists(f"{output_folder}{counter}"):
            counter += 1
        output_folder = f"{output_folder}{counter}"
        os.makedirs(output_folder)

    for i, file_path in enumerate(files, start=1):
        file_name = file_path.name
        print(f"Processing file {i}/{len(files)}: {file_name}")
        result = await Embed_any(str(file_path), openai_api_key, NLP_api_key, deepgram_api_key, language)
        if result is not None:
            with open(os.path.join(output_folder, f"{file_name}_result.json"), 'w') as f:
                json.dump(result, f, indent=6)
            print(f"Result saved to '{output_folder}/{file_name}_result.json'")
        else:
            print(f"Error: No handler could process the file {file_name}.")

# Run code 
if __name__ == "__main__":
    # Get these from your environment
    folder_path = "memory folder"
    NLP_api_key = "NLP_api_key"
    openai_api_key = "openai_api_key"
    deepgram_api_key = "deepgram_api_key"
    language = "en"

    asyncio.run(main(folder_path, openai_api_key, NLP_api_key, deepgram_api_key, language))

async def load_json_files(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                json_data = json.load(file)
                content = json_data.get("content")
                if content:
                    for item in content.values():
                        content_class = item.get("class")
                        if content_class == "table":
                            table_text = item.get("text", [])
                            table_english_text = item.get("english text", [])
                            table_embed = item.get("embed", [])
                            for i in range(len(table_text)):
                                data.append([table_text[i], table_english_text[i], table_embed[i]])
                        else:  # Treat other classes as "text"
                            text = item.get("text", "")
                            english_text = item.get("english text", "")
                            embed = item.get("embed", [])
                            data.append([text, english_text, embed])
                            
    return data

async def similarity_search(query_embed, data, top_k = 10):
    similarities = []
    query_embed = np.array(query_embed).reshape(1, -1)  # Reshape query embed to match dimensions
    for item in data:
        embed = item[-1]
        embed = np.array(embed).reshape(1, -1)  # Reshape embed to match dimensions
        similarity = cosine_similarity(query_embed, embed)[0][0]  # Calculate cosine similarity
        similarities.append(similarity)
    
    # Sort the indices based on similarity scores in descending order
    sorted_indices = np.argsort(similarities)[::-1]
    
    top_results = []
    for index in sorted_indices[:top_k]:  # Get the top 10 results
        top_results.append(data[index][0])
    
    return top_results



