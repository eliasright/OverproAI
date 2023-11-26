# OpenAI

## Quick Overview
Extremely old project so I've forgotten how to run it now. But should be in embed_all.

The project is to embed all type of files in all languages for semantic search and give AI like chatgpt to use for queries.
- Talk to a FAQ of a company
- Talk to a book
- Talk to a video

## Steps

1. Take in most file types from text, images, audio, video, and website ~ 26
2. Can understand around 50 languages given by language_schema.json
3. Using smart vector embedding where text are split based on context rather than length
4. Data in every filetypes may be embedded in 3 of the follow ways
   1. Text - Text will be split based on context, paragraph, and length. Very difficult across many languages.
   2. Table (dataframe) - each rows are embedded to be self contain. Each row having a text of their own and split like text if required
   3. images (Couldn't do this fully since openAI hasn't release their multimodal GPT-4 yet, possible now)
5. Using semantic search of the created memeory through embedding in step 3 and 4 to answer any queries.

Code was optimized for the most information retrieval using semantic search while being mindful of tokens for the AI (cost).
