# from langchain_community import document_loaders
import torch
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
import pandas as pd
from transformers import pipeline
import numpy as np
from tqdm import tqdm
load_dotenv()


print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "No GPU")

books = pd.read_csv('books_cleaned.csv')

# print(books.head())

# print(books['taged_description'])

books['taged_description'].to_csv(
    'books.txt', sep='\n', index=False, header=False)


# Use utf-8 encoding to avoid UnicodeDecodeError
raw_document = TextLoader('books.txt', encoding='utf-8').load()
# By setting chunk size as 0 we are telling the splitter to focus more on the text_splitter not on the chunk so that it split everytime from new line not from the chunkSize
text_splitter = CharacterTextSplitter(
    chunk_size=0, chunk_overlap=0, separator='\n')

# Splitting the text using splitter

document = text_splitter.split_documents(raw_document)
# print(document[0])


# creating a vector database
db_database = Chroma.from_documents(
    document, embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"),
)

query = "Give me books about love and war "
result = db_database.similarity_search(query, k=4)
# print(result)

# Now we are getting the result but we want book name not the description


def retrieve_sementic_query(
    query: str,
    top_k: int = 10
) -> pd.DataFrame:
    recs = db_database.similarity_search(query, k=50)
    books_list = []
    for i in range(0, len(recs)):
        # Extract ISBN, remove any non-digit characters (like ':')
        isbn_str = recs[i].page_content.strip('"').split()[0].replace(':', '')
        books_list += [int(isbn_str)]

    return books[books['isbn13'].isin(books_list)].head(top_k)


ans = retrieve_sementic_query("A book to teach children about nature ")

# print(ans)

#  Now we are getting the answers but we want to use our category column to make it better
books['categories'].value_counts().reset_index()
# we want only those categories which have more than 50 entries
res = books['categories'].value_counts().reset_index().query('count>50')
print(res)

# Mapping categories
category_mapping = {
    'Fiction': 'Fiction',
    'Juvenile Fiction': 'Children_fiction',
    'Biography & Autobiography': 'Non_fiction',
    'History': 'Non_fiction',
    'Literary Criticism': 'Non_fiction',
    'Philosophy': 'Non_fiction',
    'Religion': 'Non_fiction',
    'Comics & Graphic Novels': 'Fiction',
    'Religion': 'Non_fiction',
    'Drama': 'Non_fiction',
    'Juvenile Nonfiction': 'Children_nonfiction',
    'Science': 'Non_fiction',
    'Poetry': 'Fiction',
    'Business & Economics': 'Non_fiction',
    'Literary Collections ': 'Non_fiction'
}

books['simple_categories'] = books['categories'].map(category_mapping)

print(books.head())

print(books[~(books['simple_categories'].isna())])


# Now using zero shot classification to solve the problem to classify our movies into category
fiction_categories = ['Fiction', 'Non_fiction']
pipe = pipeline("zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=1)

sequence = books.loc[books['simple_categories'] == 'Fiction',
                     'description'].reset_index(drop=True)[0]

# Now we will just pass our sequence into our model so that it can classify our output
res = pipe(sequence, fiction_categories)

print(res)

max_index = np.argmax(pipe(sequence, fiction_categories)['scores'])
print(max_index)

# now using the model to ---------

max_label = pipe(sequence, fiction_categories)['labels'][max_index]
print(max_label)

# Creating a function for these tasks


def predictions(sequence, fiction_categories):
    max_index = np.argmax(pipe(sequence, fiction_categories))
    max_label = pipe(sequence, fiction_categories)['labels'][max_index]
    return max_label


actual_cats = []
predicted_cats = []

for i in tqdm((range(0, 300))):
    sequence = books.loc[books['simple_categories'] ==
                         "Fiction", "description"].reset_index(drop=True)[i]
    predicted_cats += [predictions(sequence, fiction_categories)]
    actual_cats += ["Fiction"]


for i in tqdm((range(0, 300))):
    sequence = books.loc[books['simple_categories'] ==
                         "Non_fiction", "description"].reset_index(drop=True)[i]
    predicted_cats += [predictions(sequence, fiction_categories)]
    actual_cats += ["Non_fiction"]


predictions_df = pd.DataFrame(
    {"actual_categories": actual_cats, "predicted_categories": predicted_cats})


print(predictions_df.head())

predictions_df["corr_prediction"] = np.where(
    predictions_df["actual_categories"] == predictions_df["predicted_categories"], 1, 0
)

acc = predictions_df["corr_predictions"].sum()/len(predictions_df)
print(acc)

# Now we will check only those where simple_categories is missing and we will extract the isbn so that we can merge them later
isbns = []
predicted_cats = []

missing_cats = books.loc[books["simple_categories"].isna(
), ["isbn13", "description"]].reset_index(drop=True)

print(missing_cats)

for i in tqdm(range(0, len(missing_cats))):
    sequence = missing_cats['description'][i]
    predicted_cats += [predictions(sequence, fiction_categories)]
    isbns += [missing_cats['isbns13'][i]]

missing_predicted_df = pd.DataFrame(
    {"isbn13": isbns, "predicted_categories": predicted_cats})

print(missing_predicted_df)

books = pd.merge(books, missing_predicted_df, on='isbn13', how='left')

books['simple_categories'] = np.where(books['simple_categories'].isna(
), books['predicted_categories'], books['simple_categories'])

books = books.drop(columns=['predicted_categories'])
# Checking for how many row in the other categories
books[books['categories'].str.lower().isin([
    "romance",
    "science fiction",
    "scifi",
    "fantasy",
    "horror",
    "mystery",
    "thriller",
    "comedy",
    "crime",
    "historical",

])]

books.to_csv('books_with_categories.csv', index=False)
