import openai
import numpy as np
import json
import re
import os
import nltk
import asyncio
import more_itertools

# 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

def get_book_blocks(book_content, block_size=512):
    
    tokenized_book = nltk.word_tokenize(book_content)
    book_blocks = [" ".join(tokenized_book[i:i+block_size]) for i in range(0, len(tokenized_book), block_size)]
    
    return book_blocks

def clean_book_blocks(book_blocks):
    
    # remove empty blocks
    book_blocks = [block for block in book_blocks if block.strip() != ""]
    
    # remove blocks with less than 20 tokens
    book_blocks = [block for block in book_blocks if len(nltk.word_tokenize(block)) > 20]
    
    # remove special characters such as \n, \t, \r
    book_blocks = [block.replace("\n", " ").replace("\t", " ").replace("\r", " ") for block in book_blocks]
    
    # remove multiple spaces
    book_blocks = [re.sub(' +', ' ', block) for block in book_blocks]
    
    # make lowercase
    book_blocks = [block.lower() for block in book_blocks]
    
    return book_blocks

def block_word_count(book_blocks):
    
    word_count = [len(re.findall(r'\w+', block)) for block in book_blocks]
    
    return word_count

def block_token_count(book_blocks):
    
    token_size = [len(block.split()) for block in book_blocks]
    
    return token_size

async def summarize_block(api_key,
                          block,
                          max_tokens=200,
                          temperature=0.5,
                          top_p=1.0,
                          frequency_penalty=0.0,
                          presence_penalty=0.0):
    
    openai.api_key = api_key
    
    completions = openai.Completion.create(
      engine="text-davinci-003",
      prompt=f"Summarize the following chunk of a book and keep the essence of the author. Keep in mind that there are multiple chunks and they are being fed sequentially: {block}",
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
      presence_penalty=presence_penalty,
      stop=["\n", " #"]
    )
    
    response = completions.choices[0].text
    tokens_used = len(response.split())
    
    return (response, tokens_used)

async def summarize_book(book_content, 
                   block_size=2000, 
                   max_tokens=200, 
                   temperature=0.5, 
                   top_p=1.0, 
                   frequency_penalty=0.0, 
                   presence_penalty=0.0):
    
    book_blocks = get_book_blocks(book_content)
    cleaned_book_blocks = clean_book_blocks(book_blocks)
    book_summary = ""
    
    # Split the cleaned_book_blocks into batches of 5 blocks each
    batches = more_itertools.chunked(cleaned_book_blocks, 5)
    
    for batch in batches:
        tasks = [asyncio.ensure_future(summarize_block(block)[0]) for block in batch]
        responses = await asyncio.gather(*tasks)
        
        for response in responses:
            book_summary += response
    
    return book_summary


def chat_with_author(api_key,
                     model,
                     author_bio, 
                     book_summary,
                     query,
                     max_tokens=150, 
                     temperature=0.5):
    
    openai.api_key = api_key
    
    sys_prompt = f"You are a revered author and researcher. According to your biography, you are {author_bio}." \
                + "You are now being interviewed by followers and fans of your work. Reply in a way that is consistent with your biography, personality, and writing style." 
    
    prompt = f"Based on your biography and your book/paper summary {book_summary}" \
       + f"Adopt the personality of the author and be capable of answering questions about the book. " \
       + f"For example, if the question is 'What is the name of the book?', the answer should be the name of the book. " \
       + f"Another example is 'What is the name of the author?', the answer should be the name of the author. " \
       + f"Another example is 'What is the book about?', the answer should be the summary of the book. " \
       + f"Finally, other examples can be more specific about the book's content. " \
       + "The important thing is to converse as if you are the author of the book. " \
       + "[Question]" + f"{query}" + "[Answer]" + " "
       
    if model == "davinci":
        completions = openai.Completion.create(engine="text-davinci-003", 
                                            prompt=sys_prompt + prompt, 
                                            max_tokens=max_tokens, 
                                            n=1,
                                            stop=None,
                                            temperature=temperature)

        response = completions.choices[0].text
        # count tokens used
        tokens_used = len(response.split())
        
    elif model == "gpt-4":
        completions = openai.ChatCompletion.create(model='gpt-4',
                                                   messages=[{'role':'system', 'content':sys_prompt},
                                                             {'role':'user', 'content':prompt}],
                                                   max_tokens=max_tokens,
                                                   temperature=temperature)

        response = completions.choices[0].message.content
        tokens_used = completions.usage.total_tokens
        
    return (response, tokens_used)