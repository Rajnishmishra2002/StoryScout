# StoryScout - A Smarter Way to Find Your Next Read
                                                                  “Where stories and semantics meet”
                                                                  
Welcome to StoryScout, a personal project built to explore how Large Language Models (LLMs) and modern NLP tools can make book recommendations more human. Instead of relying on ratings or keywords, StoryScout understands the meaning, tone, and emotion behind both books and your queries, like a well-read friend who gets what you’re in the mood for.

This is more than a book recommender. It's an experiment in semantic search, zero-shot classification, emotion analysis, and a user-friendly website, all wrapped up in a clean Gradio interface.

 # What This Project Does
   ** 1. Semantic Search**
        Search books by ideas and themes (e.g., "a story about revenge and redemption"), not just titles or tags.
    
   ** 2. Genre Classification**
        No hard-coded genres here. Chapterly uses zero-shot classification to decide if a book is fiction or nonfiction based on its description.
    
   ** 3. Emotion & Tone Analysis**
        Find books based on how they make you feel — joyful, tragic, suspenseful, reflective. All done using LLM-based sentiment/emotion analysis.
    
    4. Interactive App with Gradio
        All features are tied together in a simple, clean Gradio dashboard where users can explore, search, and filter books.
    

# Tech Stack
    This project is built in Python 3.11 using the following tools:
    
    HuggingFace Transformers – For embeddings, zero-shot classification, and sentiment analysis
    
    LangChain – For working with LLM tools and chaining responses
    
    ChromaDB – For fast vector search
    
    Gradio – To build a simple and clean web UI
    
    Pandas, Seaborn, Matplotlib – For data exploration
    
    KaggleHub – To easily download book datasets
    
    Python-Dotenv – To securely manage API keys


# Features In Action
  Here’s what you can do in the app:

 ** Search:**
  
  “a book about a girl who escapes a cult”
  “an emotional memoir about grief and healing”
  “a dark, suspenseful thriller with twists”
  
 ** Filter:**
  
  Fiction / Non-Fiction
  
  Emotions: Joy, Sadness, Suspense, Anger, etc.
  
**  View:**
  
  Title, author, description, tone scores, genre label
  
  Most relevant matches based on the embedding search


 # Behind the Scenes: How It Works

**Embeddings**: Each book summary is converted into a vector using OpenAI’s embedding model.

**Vector Search:** ChromaDB stores the vectors, allowing semantic search by comparing cosine similarity.

**Zero-Shot Classification:** Using a pre-trained model from HuggingFace, the system labels books as fiction or nonfiction with no training data.

**Emotion Analysis:** LLMs are prompted to describe the emotional tone of each book, and we map that back to common emotions using keyword extraction and scoring.

# Screenshots
![image](https://github.com/user-attachments/assets/203deba3-671c-40ca-ba3c-cb7b48ea3faf)
