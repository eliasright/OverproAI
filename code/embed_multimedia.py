import asyncio
import json
import os
from typing import Any, Dict
from deepgram import Deepgram
from embedding import language_embed, generate_embed_openai
from translation import NLPTranslation
import openai
import re
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi
from pydub import AudioSegment
import string
import random
import time
from moviepy.editor import VideoFileClip
import ffmpeg
import pysubs2

# MAIN
async def Embed_multimedia(file_path, deepgram_api_key, NLP_api_key, openai_api_key, language):
    # Define the permitted audio file types for different services
    whisper_permitted_types = ['.m4a', '.mp3', '.webm', '.mpga', '.wav', '.mpeg']
    deepgram_permitted_types = ['.mp3', '.mp4', '.mp2', '.aac', '.wav', '.flac', '.pcm', '.m4a', '.ogg', '.opus', '.webm']

    # Define the permitted video file types and Youtube links
    video_permitted_types = ['.mkv', '.mp4', '.avi', '.mov', '.wmv']
    youtube_links = ['https://www.youtube.com/', 'http://www.youtube.com/', 'https://youtu.be/',
                     'http://youtu.be/', 'youtu.be/', 'youtube.com/']

    # Get the file extension
    _, extension = os.path.splitext(file_path)
    
    # If it's a Youtube link
    if any(file_path.startswith(link) for link in youtube_links):
        return await embed_youtube(file_path, language, deepgram_api_key, NLP_api_key, openai_api_key)

    # If it's an audio file
    if extension in whisper_permitted_types or extension in deepgram_permitted_types:
        return await embed_audio(file_path, deepgram_api_key, NLP_api_key, openai_api_key, language)

    # If it's a video file
    elif extension in video_permitted_types:
        return await embed_video(file_path, language, deepgram_api_key, NLP_api_key, openai_api_key)

    else:
        return None

#######################################################################################################################################
################################################### For dictionary Structure ##########################################################
#######################################################################################################################################

async def create_text_content(start: float, end: float, text: str, english_text: str, embed: Any) -> Dict[str, Any]:
    return {
        'class': "text",
        
        # Classification
        'start': start,
        'end': end,
        
        # For semantic search
        'text': text,
        'english text': english_text, # Only exist if embed/ai language is in English
        'embed': embed
    }

async def create_result(file_name: str, file_extension: str, language: str, embed_language: str, content, result_type: str = "audio") -> Dict[str, Any]:
    return {
        'class': result_type,

        # file name also include its file_extension
        'file name': file_name,
        'file extension': file_extension,
        
        # Language for encoding and decoding the embed text, and ai language is what language can the text pass through (default nothing as we didnt check here)
        'language': {'prefix:': language, 'ai language': "",'embed language': embed_language},
        
        # text
        'content': content
        }
    
#######################################################################################################################################
################################################### For Deepgram Transcript ###########################################################
#######################################################################################################################################

async def deepgram_transcribe(file_path: str, language: str, deepgram_api_key: str, NLP_api_key: str, openai_api_key: str) -> Dict:
    deepgram_model = await check_audio_tier(language)
    file_extension = os.path.splitext(file_path)[1][1:]  # Extracts file extension
    MIMETYPE = f'audio/{file_extension}'
    # Initializes the Deepgram SDK
    dg_client = Deepgram(deepgram_api_key)
    file_size = os.path.getsize(file_path)
    content = {}
    
    SPLIT_LENGTH = 60 * 30
    MAX_FILE_SIZE = 2147483648

    if file_size > MAX_FILE_SIZE:
        audio_chunks = split_audio(file_path, SPLIT_LENGTH, MAX_FILE_SIZE)

        for i, chunk in enumerate(audio_chunks):
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}") as temp_file:
                # Export the chunk to the temporary file
                chunk.export(temp_file.name, format=file_extension)
                chunk_content = await process_audio_chunk(temp_file.name, deepgram_model, language, NLP_api_key,
                                                          openai_api_key, i * SPLIT_LENGTH, dg_client,
                                                          file_extension, MIMETYPE)
                # Update IDs in chunk_content and add to content
                max_id = max(content.keys(), default=0)
                chunk_content = {max_id + 1 + key: value for key, value in chunk_content.items()}
                content.update(chunk_content)

    else:
        chunk_content = await process_audio_chunk(file_path, deepgram_model, language, NLP_api_key, openai_api_key, 0,
                                                  dg_client, file_extension, MIMETYPE)
        # Update IDs in chunk_content and add to content
        max_id = max(content.keys(), default=0)
        chunk_content = {max_id + 1 + key: value for key, value in chunk_content.items()}
        content.update(chunk_content)
    
    result_type = "audio"
    # FOR VIDEO ONLY, PIGGY BACK CODE
    if "_overproAITemporary_" in file_path:
        parts = file_path.split("_overproAITemporary_")
        file_path, file_extension = parts[0], parts[0].split(".")[-1]
        result_type = "video"

    # Create the final result
    result = await create_result(os.path.basename(file_path), file_extension, language, 'en', content, result_type)

    return result

def generate_random_string(length):
    # Generate a random string of the specified length
    letters = string.ascii_lowercase + string.digits
    random_string = ''.join(random.choice(letters) for _ in range(length))
    return random_string

async def process_audio_chunk(file_path, deepgram_model, language, NLP_api_key, openai_api_key, time_offset, dg_client, file_extension, MIMETYPE):
    with open(file_path, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': MIMETYPE}
        options = {"punctuate": True, "model": deepgram_model, "language": language, "numerals": False, "utterances": True}

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            try:
                response = dg_client.transcription.sync_prerecorded(source, options)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:  # i.e. not the last attempt
                    print(f"Attempt {attempt+1} failed, retrying...")
                    continue
                else:  # last attempt
                    print(f"Attempt {attempt+1} failed, giving up.")
                    raise 

        table_response = await combine_deepgram_transcript(response, max_char=1000)

        content = {}

        for current_id, current_data in table_response.items():
            current_text = current_data["text"]
            current_start = current_data["start"] + time_offset  # Adding the offset
            current_end = current_data["end"] + time_offset  # Adding the offset
            
            language_translate = await language_embed(language)
            
            if language_translate:
                translated_text = await NLPTranslation(current_text, NLP_api_key, language, "en")
                english_text = translated_text["translated text"]
                embedding = await generate_embed_openai(translated_text, openai_api_key, "en")
            else:
                english_text = ""
                embedding = await generate_embed_openai(current_text, openai_api_key, language)
                
            embedding = embedding["embed"]
            
            content[current_id] = await create_text_content(current_start, current_end, current_text, english_text, embedding)
        
        return content

async def combine_deepgram_transcript(data, max_char):
    table = {}

    # Temporary variables
    current_text = ""
    current_start = 0
    current_end = 0
    current_id = 0
    utterances = data["results"]["utterances"]
    utterance_length = len(utterances)

    for i in range(utterance_length):
        start = utterances[i]["start"]
        end = utterances[i]["end"]
        transcript = utterances[i]["transcript"]

        # Check if the next utterance forms a complete sentence with the current one
        if i < utterance_length - 1:
            next_transcript = utterances[i+1]["transcript"]
            combined_text = (transcript + next_transcript).strip()
            current_lenght = len(current_text.encode('unicode_escape'))
            total_length = len(combined_text.encode('unicode_escape'))

            if re.match('.*[.!?](?!\d)\s*$', combined_text) and total_length <= max_char:
                # If it forms a complete sentence and doesn't exceed max_char
                # update the current text, end time and skip the next utterance
                current_text = combined_text + " "
                current_end = utterances[i+1]["end"]
                i += 1  # Skip next utterance
            elif len((current_text + transcript).encode('unicode_escape')) <= max_char:
                # If the current transcript doesn't exceed max_char
                # add it to current text
                current_text += transcript + " "
                current_end = end
            else:
                # If current transcript exceeds max_char
                # store the current text and start a new one
                table[current_id] = {"text": current_text.strip(), "start": current_start, "end": current_end}
                current_id += 1
                current_text = transcript + " "
                current_start = start

        else:
            # For the last utterance, just add the transcript to the current text
            current_text += transcript + " "
            current_end = end

    # Add the last text to the table
    table[current_id] = {"text": current_text.strip(), "start": current_start, "end": current_end, "char_length": len(current_text.encode('unicode_escape'))}
    return table


async def check_audio_tier(language_code):
    # Open and load the JSON file
    with open('language_schema.json') as f:
        data = json.load(f)

    # Access the "audio tier" field for the input language
    audio_tier = data.get(language_code, {}).get('audio tier')

    # Return True if the "audio tier" field exists, False otherwise
    return audio_tier

#######################################################################################################################################
################################################### For Whisper Transcript ############################################################
#######################################################################################################################################

async def whisper_transcribe_translate(file_path, openai_api_key, NLP_api_key, language, model="whisper-1", response_format="json"):
    try:
        prompt = "Hello, welcome to my lecture."
        openai.api_key = openai_api_key
        MAX_FILE_SIZE = 26214400
        transcript = ''
        text_contents = {}
        start_time = end_time = 0

        # Check file size
        if os.path.getsize(file_path) > MAX_FILE_SIZE:
            audio_chunks = split_audio(file_path, 10 * 60, MAX_FILE_SIZE)
            if not audio_chunks:
                return None
            
            for chunk in audio_chunks:
                with tempfile.NamedTemporaryFile(suffix=".mp3") as temp_file:
                    chunk.export(temp_file, format='mp3')

                    with open(temp_file, "rb") as audio_file:
                        transcript_data = openai.Audio.translate(model=model, file=audio_file, response_format=response_format, prompt=prompt) if language.lower() not in ["english", "en"] else openai.Audio.transcribe(model=model, file=audio_file, response_format=response_format, prompt=prompt)
                        transcript += transcript_data["text"]

        else:
            with open(file_path, "rb") as audio_file:
                transcript_data = openai.Audio.translate(model=model, file=audio_file, response_format=response_format, prompt=prompt) if language.lower() not in ["english", "en"] else openai.Audio.transcribe(model=model, file=audio_file, response_format=response_format, prompt=prompt)
                transcript = transcript_data["text"]

        sections = await break_paragraph(transcript, 1000)

        for i, section in enumerate(sections):
            try:
                embed_data = await generate_embed_openai(section, openai_api_key, language)
                embed = embed_data["embed"]
                translation_data = await NLPTranslation(section, NLP_api_key, "en", language)
                translated_text = translation_data["translated text"]
                english_text = "" if language.lower() in ["english", "en"] else section
                text_content = await create_text_content(start_time, end_time, translated_text, english_text, embed)
                text_contents[i] = text_content
            except Exception as e:
                print(f"Error processing section {i}: {str(e)}")
                return None

        result_type = "audio"
        # FOR VIDEO ONLY, PIGGY BACK CODE
        if "_overproAITemporary_" in file_path:
            parts = file_path.split("_overproAITemporary_")
            file_path, file_extension = parts[0], parts[0].split(".")[-1]
            result_type = "video"

        try:
            result = await create_result(file_path, file_extension, language, 'en', text_contents, result_type)
        except Exception as e:
            print(f"Error in create_result: {str(e)}")
            return None

        return result

    except Exception as e:
        print(f"Error in whisper_transcribe_translate: {str(e)}")
        return None
    
def convert_bytes_to_megabytes(size_in_bytes):
    return size_in_bytes / (1024 * 1024)

def get_audio_duration(file_size, bit_rate=128):
    bit_rate = bit_rate * 1000
    file_size = file_size * 8
    duration = file_size / bit_rate
    return duration

def split_audio(file_path, split_length=60*30, MAX_CONTENT_SIZE=2147483648):
    try:
        audio = AudioSegment.from_file(file_path)

        audio_chunks = []

        start_time = 0
        end_time = split_length * 1000
        chunk_counter = 1

        while start_time < len(audio):
            end_time = min(end_time, len(audio))  # Making sure end_time doesn't exceed the audio length

            # Get the chunk and add it to the list
            chunk = audio[start_time:end_time]

            # Create a temporary file
            fd, path = tempfile.mkstemp(suffix=".mp3")

            try:
                # Export the chunk to the temporary file
                chunk.export(path, format="mp3")
                file_size = os.path.getsize(path)

                # If the size exceeds the limit, adjust the split length
                while file_size > MAX_CONTENT_SIZE:
                    print(file_size, MAX_CONTENT_SIZE)
                    print("Chunk size is too large, adjusting split length...")
                    
                    # Decrease the split length proportionally to the size difference
                    size_ratio = file_size / MAX_CONTENT_SIZE
                    split_length = int(split_length / size_ratio)
                    end_time = start_time + split_length * 1000  # Update end_time accordingly
                    
                    chunk = audio[start_time:end_time]
                    chunk.export(path, format="mp3")
                    file_size = os.path.getsize(path)

                # At this point, chunk size is guaranteed to be within the limit
                audio_chunks.append(chunk)
                print(f"Created chunk with duration: {len(chunk)/1000.0} seconds")
                print(f"Chunk {chunk_counter}: Start time = {start_time/1000.0}s, End time = {end_time/1000.0}s, Duration = {len(chunk)/1000.0}s, Size = {os.path.getsize(path)} bytes")

            finally:
                # Close the file descriptor and remove the temporary file
                os.close(fd)
                os.remove(path)

            start_time = end_time
            end_time += split_length * 1000
            chunk_counter += 1

        return audio_chunks
    except Exception as e:
        print(f"Error occurred while splitting audio: {e}")

async def break_paragraph(paragraph, max_char):
    sentences = re.split(r'(?<=[.!?])\s+(?!\d)', paragraph)  # Split paragraph into sentences, excluding periods followed by a number
    mini_paragraphs = []
    current_paragraph = ""

    for sentence in sentences:
        if len(current_paragraph) + len(sentence) + 1 <= max_char:
            current_paragraph += sentence + " "  # Add a space after the sentence
        else:
            mini_paragraphs.append(current_paragraph.strip())
            current_paragraph = sentence + " "  # Start a new paragraph

    if current_paragraph:
        mini_paragraphs.append(current_paragraph.strip())

    return mini_paragraphs

#######################################################################################################################################
################################################ For main transcription ###############################################################
#######################################################################################################################################

# Check the audio services for a given language
async def check_audio_services(language_code):
    # Open and load the JSON file
    with open('language_schema.json') as f:
        data = json.load(f)

    # Access the "audio transcribe" field for the input language
    audio_transcribe = data.get(language_code, {}).get('audio transcribe')

    # Return the list of services if the "audio transcribe" field exists, None otherwise
    return audio_transcribe

# Check if the file is an audio file
def is_whisper_audio_file(file_path):
    # Define the permitted audio file types
    permitted_types = ['m4a', 'mp3', 'webm', 'mpga', 'wav', 'mpeg']

    # Get the file extension
    _, extension = os.path.splitext(file_path)

    # Check if the file extension is in the list of permitted types
    return extension.lstrip('.').lower() in permitted_types

def is_deepgram_audio_file(file_path):
    # Define the permitted audio file types
    permitted_types = ['mp3', 'mp4', 'mp2', 'aac', 'wav', 'flac', 'pcm', 'm4a', 'ogg', 'opus', 'webm']

    # Get the file extension
    _, extension = os.path.splitext(file_path)

    # Check if the file extension is in the list of permitted types
    return extension.lstrip('.').lower() in permitted_types

async def embed_audio(file_path, deepgram_api_key, NLP_api_key, openai_api_key, language):
    services = await check_audio_services(language)

    if services is None:
        print(f"No audio transcribe service found for language: {language}")
        return

    output = {}

    if language.lower() == "en" or language.lower() == "english":
        # Check if the file is a valid audio file for Deepgram
        if not is_deepgram_audio_file(file_path):
            print(f"Invalid file type for Deepgram: {file_path}")
            return
        output = await deepgram_transcribe(file_path, language, deepgram_api_key, NLP_api_key, openai_api_key)
    elif "whisper" in services:
        # Check if the file is a valid audio file for Whisper
        if not is_whisper_audio_file(file_path):
            print(f"Invalid file type for Whisper: {file_path}")
            return
        output = await whisper_transcribe_translate(file_path, language, deepgram_api_key, NLP_api_key, openai_api_key)
    elif "deepgram" in services and language != "en":
        # Check if the file is a valid audio file for Deepgram
        if not is_deepgram_audio_file(file_path):
            print(f"Invalid file type for Deepgram: {file_path}")
            return
        output = await deepgram_transcribe(file_path, language, deepgram_api_key, NLP_api_key, openai_api_key)
        
    return output

#######################################################################################################################################
################################################### For embed video #############################################################
#######################################################################################################################################

async def embed_video(file_path, language, deepgram_api_key, NLP_api_key, openai_api_key):
    file_extension = os.path.splitext(file_path)[1]
    
    # YOUTUBE
    if file_path.startswith('https://www.youtube.com/') or file_path.startswith('https://youtu.be/'):
        return await embed_youtube(file_path, language, deepgram_api_key, NLP_api_key, openai_api_key)
    
    # FILES
    if file_extension == '.mkv':
        attempt = await embed_mkv(file_path, language, NLP_api_key, openai_api_key)
        
        if attempt is None:
            return await transcribe_video(file_path, deepgram_api_key, NLP_api_key, openai_api_key, language)
        return attempt
    elif file_extension == '.mp4':
        return await transcribe_video(file_path, deepgram_api_key, NLP_api_key, openai_api_key, language)
    elif file_extension == '.avi':
        return await transcribe_video(file_path, deepgram_api_key, NLP_api_key, openai_api_key, language)
    elif file_extension == '.mov':
        return await transcribe_video(file_path, deepgram_api_key, NLP_api_key, openai_api_key, language)
    elif file_extension == '.wmv':
        return await transcribe_video(file_path, deepgram_api_key, NLP_api_key, openai_api_key, language)
    else:
        raise ValueError("Unsupported file format: {}".format(file_extension))

async def transcribe_video(file_path, deepgram_api_key, NLP_api_key, openai_api_key, language):
    video = VideoFileClip(file_path)
    suffix = ".mp3"
    
    with tempfile.NamedTemporaryFile(suffix=suffix, prefix=f"{file_path}_overproAITemporary_", delete=False) as temp_audio:
        temp_filename = temp_audio.name
        
        video.audio.write_audiofile(temp_filename)
        time.sleep(0.5)
        
        result = await embed_audio(temp_filename, deepgram_api_key, NLP_api_key, openai_api_key, language)
        
        os.remove(temp_filename)
    
    return result

async def embed_mkv(file_path, language, NLP_api_key, openai_api_key, max_char=1000):
    try:
        temp_dir = tempfile.TemporaryDirectory()
        out_file = os.path.join(temp_dir.name, f"{generate_random_string(8)}output.srt")
        
        ffmpeg.input(file_path).output(out_file, map='0:s:0').run()
        subs = pysubs2.load(out_file, encoding="utf-8")

        # Initialize the transcript table and temporary variables
        table = {}
        current_text = ""
        current_start = 0
        current_end = 0
        current_id = 0

        for sub in subs:
            text = sub.text
            text = re.sub(r'{\\b[01]}', '', text)  # Removes the bold control codes
            text = re.sub(r'{\\i[01]}', '', text)  # Removes the italic control codes
            text = text.replace("\\N", " ")  # Replaces \N with a space

            # Calculate the start and end times
            start = sub.start / 1000.0  # assuming the time is in milliseconds
            end = sub.end / 1000.0  # assuming the time is in milliseconds

            # Calculate unicode byte length of the combined current and new transcript
            total_length = len((current_text + text).encode('unicode_escape'))

            # Check if the combined transcript exceeds the maximum allowed character limit
            if total_length <= max_char:
                # If it doesn't, update the current text, and the current end time
                current_text += text + " "
                current_end = end
            else:
                # If it does, store the current text and start a new one
                table[current_id] = {"text": current_text.strip(), "start": current_start, "end": current_end}
                current_id += 1
                current_text = text + " "
                current_start = start

        # Add the last text to the table
        table[current_id] = {"text": current_text.strip(), "start": current_start, "end": current_end, "char_length": len(current_text.encode('unicode_escape'))}
        
        text_contents = {}

        for current_id, current_data in table.items():
            current_text = current_data["text"]
            current_start = current_data["start"]  # Adding the offset
            current_end = current_data["end"]  # Adding the offset

            language_translate = await language_embed(language)

            if language_translate:
                translated_text = await NLPTranslation(current_text, NLP_api_key, language, "en")
                english_text = translated_text["translated text"]
                embedding = await generate_embed_openai(translated_text, openai_api_key, "en")
            else:
                english_text = ""
                embedding = await generate_embed_openai(current_text, openai_api_key, language)

            embedding = embedding["embed"]

            text_contents[current_id] = await create_text_content(current_start, current_end, current_text, english_text, embedding)

        file_name, file_extension = os.path.splitext(file_path)

        result = await create_result(file_path, file_extension, language, 'en', text_contents, "video")

        # Delete the temporary file
        if os.path.exists(out_file):
            os.remove(out_file)

        return result
    except ffmpeg.Error as e:
        return None
    
#######################################################################################################################################
################################################### For youtube embed #################################################################
#######################################################################################################################################

async def embed_youtube(url, language, deepgram_api_key, NLP_api_key, openai_api_key, max_char=1000):
    # Extract the video ID from the YouTube URL
    video_id = extract_video_id(url)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Initialize the transcript table and temporary variables
        table = {}
        current_text = ""
        current_start = 0
        current_end = 0
        current_id = 0

        for segment in transcript:
            text = segment['text']

            # Calculate the start and end times
            start = segment['start']
            end = segment['start'] + segment['duration']

            # Calculate unicode byte length of the combined current and new transcript
            total_length = len((current_text + text).encode('unicode_escape'))

            # Check if the combined transcript exceeds the maximum allowed character limit
            if total_length <= max_char:
                # If it doesn't, update the current text, and the current end time
                current_text += text + " "
                current_end = end
            else:
                # If it does, store the current text and start a new one
                table[current_id] = {"text": current_text.strip(), "start": current_start, "end": current_end}
                current_id += 1
                current_text = text + " "
                current_start = start

        # Add the last text to the table
        table[current_id] = {"text": current_text.strip(), "start": current_start, "end": current_end, "char_length": len(current_text.encode('unicode_escape'))}
        
        text_contents = {}

        for current_id, current_data in table.items():
            current_text = current_data["text"]
            current_start = current_data["start"]  # Adding the offset
            current_end = current_data["end"]  # Adding the offset

            language_translate = await language_embed(language)

            if language_translate:
                translated_text = await NLPTranslation(current_text, NLP_api_key, language, "en")
                english_text = translated_text["translated text"]
                embedding = await generate_embed_openai(translated_text, openai_api_key, "en")
            else:
                english_text = ""
                embedding = await generate_embed_openai(current_text, openai_api_key, language)

            embedding = embedding["embed"]

            text_contents[current_id] = await create_text_content(current_start, current_end, current_text, english_text, embedding)

        result = await create_result(url, "youtube", language, 'en', text_contents, "video")

        output_file = "output_audio.json"

        with open(output_file, "w") as output:
            json.dump(result, output, indent=4)

        return result
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def extract_video_id(url):
    # Extract the video ID from the YouTube URL
    video_id = None
    if "youtube.com" in url:
        video_id = url.split("v=")[1]
        if "&" in video_id:
            video_id = video_id.split("&")[0]
    elif "youtu.be" in url:
        video_id = url.split("/")[-1]

    return video_id
