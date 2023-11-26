# OpenAI Project Overview

## Quick overview
This project is many months old so I have forgotten how to run it and do not intend on doing so unless I wish to continue working on it in the future. 

It aimed to enable embedding various types of files in multiple languages for semantic search purposes.

The primary objective of this project was to create a versatile tool for AI, such as ChatGPT, to perform queries on a wide range of file types. Allow user to:
- Interacting with a company's FAQ instantly without human help
- Ask a video any questions without having to go through it yourself
- Talk directly to a pdf so you can access an answer without having to go through it

## Key steps of the code

1. **File Type Handling**: The system was designed to process a variety of file types, including text, images, audio, video, and websites, totaling around 26 different formats.

2. **Multilingual Capability**: It had the ability to understand approximately 50 languages, as specified in the "language_schema.json."

3. **Smart Vector Embedding**: Utilizing a smart vector embedding technique, the project split text based on context rather than just length, making it more contextually aware. 

4. **Data Embedding Methods**: Data within various file types was embedded in one of three ways:
   - Text: Text was divided based on context, paragraphs, and length, which proved to be challenging across multiple languages.
   - Table (Dataframe): Each row of tables was embedded independently, with each row having its own text, and text splitting was applied if necessary.
   - Images: While not fully implemented due to the absence of OpenAI's multimodal GPT-4 at that time, there was potential for image embedding.

5. **Semantic Search**: The project incorporated a semantic search mechanism using the embeddings generated in steps 3 and 4. This allowed the system to respond to various queries effectively. Queries answer contain:
   1. Textual response to the answer
   2. The context the AI used to answer the query
   3. The source of the answer in the file (be it page or timestamp of a video)

The code was optimized to maximize information retrieval through semantic search while being mindful of token usage to manage AI costs efficiently. Please note that since the project is old, be aware that because this project is quite old, certain OpenAI components it relied on may no longer be supported or functional, which could affect the code's retrieval and execution
