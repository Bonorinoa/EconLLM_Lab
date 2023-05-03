from utils import summarize_book, chat_with_author

work_filename = "TheoryMoralSentiments"

with open(f"AdamSmith_MoralSentiments\\{work_filename}.txt", encoding='utf8') as f:
    work_to_summarized = f.read()
    
work_summary = summarize_book(work_to_summarized, 
                              api_key="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",
                              temperature=0.55,
                              model="gpt-4") # model here is used to compute block size

bio_filename = "AdamSmith_Bio"

with open(f"AdamSmith_MoralSentiments\\{bio_filename}.txt", encoding='utf8') as f:
    author_bio = f.read()
    
query = "What is the main idea of the book?"

response, tokens_used = chat_with_author(api_key="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",
                                         model="gpt-4",
                                         author_bio=author_bio,
                                         book_summary=work_summary,
                                         query=query,
                                         max_tokens=300,
                                         temperature=0.7)

print("------ Summary ------")
print(work_summary)

print(f"------ Response {tokens_used} ------")
print(response)