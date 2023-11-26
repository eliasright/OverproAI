import re
import textwrap

async def split_text_by_chars(text, max_chars, avoid_spaces=True):
    if not isinstance(text, str):
        raise TypeError("Expected text to be a string")
    if not isinstance(max_chars, int):
        raise TypeError("Expected max_chars to be an integer")
    if max_chars <= 0:
        raise ValueError("max_chars should be greater than zero")

    # Split the text into sections based on 'text2' criteria
    all_sections = re.split(r'(\n.{'+str(max_chars//4)+','+str(max_chars)+'}\n)', text)

    # Prepare for the segmented sections
    segmented_sections = []

    for section in all_sections:
        # If section length is less than max_chars
        if len(section) <= max_chars:
            segmented_sections.append(section)
            continue

        # Split the section into sentences based on punctuation marks
        sentences = re.split('(?<=[.!?]) +', section)

        # Process each sentence
        segments = []
        for sentence in sentences:
            print()
            print(sentence)
            # If sentence length is less than max_chars
            if len(sentence) <= max_chars:
                segments.append(sentence)
            else:
                # Split the sentence into max_chars-sized segments
                segments.extend(textwrap.wrap(sentence, max_chars))

        # Combine segments together into new section
        new_section = ""
        while segments:
            if len(new_section) + len(segments[0]) <= max_chars:
                if new_section:
                    new_section += " "
                new_section += segments.pop(0)
            else:
                segmented_sections.append(new_section.strip())
                new_section = ""

        # Add the remaining new section
        if new_section:
            segmented_sections.append(new_section.strip())

    # Combine the sections together
    combined_segments = []
    new_segment = ""
    while segmented_sections:
        if len(new_segment) + len(segmented_sections[0]) <= max_chars:
            if new_segment:
                new_segment += "\n\n"
            new_segment += segmented_sections.pop(0)
        else:
            combined_segments.append(new_segment.strip())
            new_segment = ""

    # Add the remaining new segment
    if new_segment:
        combined_segments.append(new_segment.strip())

    # Replace newline characters with a single space
    combined_segments = [re.sub('\n', ' ', segment) for segment in combined_segments]

    # If avoid_spaces is True, replace multiple consecutive spaces with a single space
    if avoid_spaces:
        combined_segments = [re.sub(' +', ' ', segment) for segment in combined_segments]

    return combined_segments

"""

import requests
from bs4 import BeautifulSoup
import re

def extract_text(url):
    # Make a request
    response = requests.get(url)
    response.raise_for_status()  # Raise an HTTPError if the status is 4xx, 5xx

    # Parse the document
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    # Get text
    text = soup.get_text()

    return text

url = "https://www.vox.com/climate/23746045/state-farm-california-climate-change-insurance-wildfire-florida-flood"
for i, a in enumerate(split_text_by_chars(extract_text(url), 1000)):
    print(i, ".", a)
    print()

"""



