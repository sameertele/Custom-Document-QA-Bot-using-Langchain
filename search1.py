import requests
import json
import torch
from bs4 import BeautifulSoup
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from langchain.chains import ConversationChain
from langchain_community.llms import HuggingFacePipeline
import streamlit as st
import time
import os
import gensim
import nltk
from nltk.summary import summarize

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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" 

# Function to fetch research papers
def fetch_research_papers(keyword, retries=3):
    url = f"http://export.arxiv.org/api/query?search_query=all:{keyword}&start=0&max_results=5"
    papers = []
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad status codes
            soup = BeautifulSoup(response.text, 'lxml')
            for entry in soup.find_all('entry'):
                title = entry.title.text
                link = entry.id.text
                papers.append({'title': title, 'link': link})
            return papers
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retrying
    return papers

# Function to fetch paper content from the arXiv link
def fetch_paper_content(arxiv_link):
    url = f"https://arxiv.org/pdf/{arxiv_link}.pdf"
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.text
        return content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching paper content: {e}")
        return None

# Function to retrieve and summarize relevant papers
def retrieve_and_summarize(papers, question):
    # Create a dictionary to store paper titles and summaries
    paper_summaries = {}
    
    # Retrieve and summarize each paper
    for paper in papers:
        # Fetch the paper content
        paper_content = fetch_paper_content(paper['link'])
        
        # If content is fetched successfully, summarize
        if paper_content:
            summary = summarize(paper_content, ratio=0.2)  # Adjust the ratio as needed
            paper_summaries[paper['title']] = summary
        else:
            print(f"Error fetching content for {paper['title']}")
    
    # Use Gensim to find relevant papers based on the question
    dictionary = gensim.corpora.Dictionary([summary for summary in paper_summaries.values()])
    bow = [dictionary.doc2bow(summary) for summary in paper_summaries.values()]
    corpus = gensim.models.tfidfmodel.TfidfModel(bow)
    query_bow = dictionary.doc2bow(question.split())
    similarities = corpus[query_bow]
    
    # Select the most relevant paper and its summary
    most_relevant_paper = max(similarities, key=lambda x: x[1])
    relevant_paper_title = paper_summaries.keys()[most_relevant_paper[0]]
    relevant_paper_summary = paper_summaries[relevant_paper_title]
    
    return relevant_paper_title, relevant_paper_summary

# Load GPT-2 Model and Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load or initialize local embeddings storage
embeddings_file = 'embeddings.json'

# Load existing embeddings from the JSON file
def load_embeddings():
    try:
        with open(embeddings_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Embed and upload documents locally
def embed_and_upload_documents(documents):
    embeddings = load_embeddings()
    for doc in documents:
        inputs = tokenizer(doc['text'], return_tensors='pt', max_length=512, truncation=True)
        with torch.no_grad():  # Avoid gradients calculation
            embeddings_vector = model(**inputs).last_hidden_state.mean(dim=1).numpy()[0].tolist()  # Convert to list
        embeddings[doc['title']] = embeddings_vector
    
    # Save embeddings to the JSON file
    with open(embeddings_file, 'w') as f:
        json.dump(embeddings, f)

# Initialize the model pipeline for text generation
gpt2_pipeline = pipeline('text-generation', model='gpt2', max_new_tokens=50, device=0)

# Initialize the HuggingFacePipeline wrapper for LangChain
huggingface_pipeline = HuggingFacePipeline(pipeline=gpt2_pipeline)

# Initialize the ConversationChain with the wrapped Hugging Face pipeline
conversation_chain = ConversationChain(llm=huggingface_pipeline)

# Streamlit app
st.title("Research Paper QA Bot")

topic = st.text_input("Enter a topic to search for research papers:")
if st.button("Search"):
    if topic:
        papers = fetch_research_papers(topic)
        if papers:
            st.write("Research Papers Found:")
            for paper in papers:
                st.write(f"- [{paper['title']}]({paper['link']})")
        else:
            st.write("No papers found or an error occurred. Please try again.")
    else:
        st.write("Please enter a topic.")

question = st.text_input("Ask a question about the topic:")
if st.button("Get Answer"):
    if question:
        # Retrieve and summarize relevant papers
        relevant_paper_title, relevant_paper_summary = retrieve_and_summarize(papers, question)
        
        # Combine the relevant paper summary and the question for the GPT-2 model
        combined_text = relevant_paper_summary + "\n" + question
        
        # Call the conversation chain for answering questions
        answer = conversation_chain.invoke({"input": combined_text, "max_new_tokens": 100})
        st.write(answer['output'])
    else:
        st.write("Please enter a question.")