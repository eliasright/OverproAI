import openai
import tiktoken

#######################################################################################################################################
############################################################## Open AI chat ###########################################################
#######################################################################################################################################

async def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

async def generate_chat_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
        
    # Split the response into chunks of 1000 characters
    return {"content":response.choices[0].message['content']} 
