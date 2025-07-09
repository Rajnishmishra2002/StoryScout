import pandas as pd
import numpy as np
from dotenv import load_dotenv
import re

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv('books_with_emotions.csv')

books['large_thumbnail'] = books['thumbnail']+"&fife=w800"
books['large_thumbnail'] = np.where(books['large_thumbnail'].isna(), "cover-not-found.jpg",
                                    books['large_thumbnail'])


raw_documents = TextLoader('books.txt', encoding='utf-8').load()
text_splitter = CharacterTextSplitter(
    separator='/n', chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, HuggingFaceEmbeddings())


def retrieve_sementic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    intial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=intial_top_k)
    books_list = []
    for rec in recs:
        print("rec.page_content:", rec.page_content)
        match = re.search(r'(\d{13})', rec.page_content)
        if match:
            books_list.append(int(match.group()))
    print("books_list:", books_list)
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)
    # print("books_list:", books_list)
    # print("books['isbn13'] dtype:", books['isbn13'].dtype)
    # print("books_recs shape after filtering:", books_recs.shape)
    # print("First few rec.page_content:", [
    #       rec.page_content for rec in recs[:5]])
    if category != "All":
        books_recs.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == 'Surprising':
        books_recs.sort_values(by='surprise', ascending=False, inplace=True)
    elif tone == 'Angry':
        books_recs.sort_values(by='angry', ascending=False, inplace=True)
    elif tone == 'Suspensful':
        books_recs.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == 'Sad':
        books_recs.sort_values(by='sadness', ascending=False, inplace=True)

    return books_recs


def recommend_books(
    query: str,
    category: str,
    tone: str
):
    recommendations = retrieve_sementic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row['description']
        truncted_desc_split = description.split()
        truncted_description = " ".join(truncted_desc_split[:30] + ["..."])

        authors_split = row['authors'].split(":")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{','.join(authors_split[:-1])},and{authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncted_description}"
        results.append((row['large_thumbnail'], caption))
    return results


# categories = ["All"]+sorted(books['simple_categories'].unique())
categories = ["All"] + sorted(books['simple_categories'].dropna().unique())

tones = ["All"] + ["Happy", "Suprising", "Angry", "Suspensful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please Enter a description of a book ",
                                placeholder='A Story about love')
        category_dropdown = gr.Dropdown(
            choices=categories, label='Select a category', value="All")
        tone_dropdown = gr.Dropdown(
            choices=tones, label="Select an emotional tone", value="All")
        submit_button = gr.Button('Find Recommendations')

    gr.Markdown("## Recommendations")

    output = gr.Gallery(label='Recommended books', columns=8, rows=2)

    submit_button.click(fn=recommend_books, inputs=[
                        user_query, category_dropdown, tone_dropdown], outputs=output)


if __name__ == "__main__":
    dashboard.launch()
