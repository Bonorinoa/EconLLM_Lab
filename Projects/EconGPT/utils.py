import openai
import numpy as np
import json
import re
import os
import nltk
import asyncio
import more_itertools

# block size must be set such that the tokens used for the book summary is less than the maximum tokens allowed for the model
def estimate_block_and_tokens(text, model='davinci'):
    
    text_tokens = len(nltk.word_tokenize(text))
    if model == 'davinci':
        model_tokens = 3000
    elif model == 'gpt-4':
        model_tokens = 7000
        
    # block size for davinci should be such that the final summary is around 3000 to 3500 tokens (4k max so we leave 1000 to 500 for prompt)
    # block size for gpt-4 should be such that the final summary is around 7000 to 7500 tokens (8k max so we leave 1000 to 500 for prompt)
    # the final summary is the sum of the tokens used for each summarized block. This is influenced by the max tokens, default set to 300.
    ## So, if the max tokens is set to 300, then the final summary will be 300 * number of blocks.
    ### In other words: max tokens = model tokens / number of blocks
    ## Therefore, the block size should be set such that the number of blocks is around 10.
    ### In other words, block size = text tokens / max_tokens
    
    max_tokens = (model_tokens / 10)
    block_size = int(text_tokens / max_tokens)
    
    return (block_size, max_tokens)

def get_book_blocks(book_content):
    
    block_size, _ = estimate_block_and_tokens(book_content)
    
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
                          temperature=0.5, # low temperature to keep the features of the original text
                          top_p=1.0,
                          frequency_penalty=0.0,
                          presence_penalty=0.0):
    
    openai.api_key = api_key
    
    _, max_tokens = estimate_block_and_tokens(block, model='davinci')
    
    completions = openai.Completion.create(
      engine="text-davinci-003",
      prompt=f"Summarize the following chunk of a scientific research work and focus on keeping the essence and style of the author. Also, make sure to identify major historical events and dates in the text. Keep in mind that there are multiple chunks and they are being fed sequentially: {block}",
      max_tokens=int(max_tokens),
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
                         api_key):
    
    openai.api_key = api_key
    
    book_blocks = get_book_blocks(book_content)
    cleaned_book_blocks = clean_book_blocks(book_blocks)
    book_summary = ""
    
    # Split the cleaned_book_blocks into batches of 5 blocks each
    batches = more_itertools.chunked(cleaned_book_blocks, 5)
    
    # Summarize each batch of blocks asyncrhonously for faster processing
    for batch in batches:
         # Create a list of tasks using summarize_block without indexing
        tasks = [await asyncio.ensure_future(summarize_block(api_key, block)) for block in batch]
        # Await the completion of all tasks and store the results
        results = await asyncio.gather(*tasks)
        
        # Extract the summaries from the results and concatenate
        for response, _ in results:
            book_summary += response
    
    # Final summary is the concatenation of all the block summaries
    return book_summary


def chat_with_author(api_key,
                     model,
                     author_bio,
                     workTitle,
                     book_summary,
                     query,
                     max_tokens=100, 
                     temperature=0.5):
    
    openai.api_key = api_key
    
    sys_prompt = f"You are a revered author and researcher. According to your biography, you are {author_bio}." \
                + "You are now being interviewed by followers and fans of your work. Reply in a way that is consistent with your biography, personality, and writing style." 
    
    prompt = f"Based on your biography and the summary {book_summary} of your work titled {workTitle}" \
        + "converse as if you are the author of the book. " \
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