from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_document(text, max_length=130, min_length=30):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

document_text = """ The Titanic was a British passenger liner that sank in the North Atlantic Ocean on 15 April 1912 after striking an iceberg 
during her maiden voyage from Southampton, UK, to New York City, United States. Of the estimated 2,224 passengers and crew 
aboard, more than 1,500 died, making the sinking one of the deadliest peacetime maritime disasters in history. 
The disaster drew public attention, provided foundational lessons for ship safety, and continues to generate research and fascination. """

summary = summarize_document(document_text)

print("Summary of the document:")
print(summary)
