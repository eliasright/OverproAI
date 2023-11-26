import requests
import asyncio

API_2_CONCURRENCY_LIMIT = 2
api_2_semaphore = asyncio.Semaphore(API_2_CONCURRENCY_LIMIT)

class APIError(Exception):
    pass

# Use NLP Translation instead of azure due to better pricing
async def NLPTranslation(text: str, NLP_api_key: str, lan_in: str, lan_out: str = "en") -> dict:
    if not isinstance(text, str) or not isinstance(NLP_api_key, str) or not isinstance(lan_in, str) or not isinstance(lan_out, str):
        raise TypeError("All parameters must be of type string.")
    if len(text) == 0:
        return {"translated text": text, "cost": 0}

    # Return input text and 0 character count since no character was used
    # Return input text if input text is the error_output to research user
    if lan_in == lan_out:
        return {"translated text": text, "cost": 0}

    async with api_2_semaphore:
        url = "https://nlp-translation.p.rapidapi.com/v1/translate"
        querystring = {"text":text,"to":lan_out,"from":lan_in}
        headers = {
        	"X-RapidAPI-Key": NLP_api_key,
        	"X-RapidAPI-Host": "nlp-translation.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers, params=querystring)
        response = response.json()

        # Initial values
        translated_text = ""
        translated_characters = ""
        cost = 0

        # This translation cost ratio is 10 dollars = 15 million
        cost_ratio = 10/15000000

        # If response is not 200, need to be more specific like azure
        if response['status'] == 200:
            translated_text = response['translated_text'][lan_out]
            translated_characters = response['translated_characters']
            cost = cost_ratio * translated_characters
        else:
            raise APIError(f"API Error with status code: {response['status']}")

        return {"translated text": translated_text, "cost": cost}
    
    
    
    