Custom Document QA Bot using Langchain

Overview

The Custom Document QA Bot is a conversational AI system designed to retrieve and answer questions based on research documents. By leveraging GPT-2 and Langchain, the bot integrates document embeddings, research paper retrieval, and natural language processing techniques to provide accurate and contextually relevant answers. The integration with the arXiv API allows users to query a wide range of research papers.

Features

Conversational AI: Utilizes GPT-2 to answer questions interactively.

Document Embedding: Implements document vectorization for efficient retrieval of relevant content.

Research Paper Retrieval: Integrates the arXiv API to access research papers directly.

Performance Optimization: Fine-tuned for low-latency responses and high-quality answers.

Requirements

Software and Libraries

Python 3.8+

Langchain

Transformers

arXiv API

Flask (for web interface)

Additional Python libraries:

torch

numpy

requests

sentence-transformers

Hardware

A GPU-enabled environment is recommended for optimal performance when running GPT-2.
