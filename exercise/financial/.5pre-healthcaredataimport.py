# %% [markdown]
# ## Load and embed your data using langchain
# 
# In this step, we want to use `OpenAIEmbeddings` to vectorize our data, so we have to get the OpenAI API Key.
# 
# The following block of code is used to manage environment variables, specifically for loading and setting the OpenAI API key. It begins by importing necessary modules for operating system interactions and secure password input. 
# 
# The script then checks if the `OPENAI_API_KEY` is already set in the environment variables. If not set, it will prompt the user to input their API key, illustrating how one could securely obtain and set this key at runtime. Using environment variables for such sensitive information, rather than hardcoding it into your application, enhances security by keeping credentials out of the source code and under strict control via environment configurations.

# %%
import getpass
import os
from dotenv import load_dotenv

load_dotenv(override=True)




# %% [markdown]
# 
# The next block imports a variety of libraries and modules for completing advanced language processing tasks. These include handling and storing documents, loading textual and JSON data, splitting text based on character count, and utilizing embeddings from OpenAI, Hugging Face, and potentially faster embedding methods. 

# %%
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.fastembed import FastEmbedEmbeddings

from langchain_iris import IRISVector


# %% [markdown]
# Next, set up the process for loading, splitting, and preparing to embed text documents from a dataset.
# 
# The first step is to initialize a JSONLoader to load documents from a specified file. The line
# `json_lines=True` specifies that we are loading files from a json_lines file, which is a file format where each line is a complete JSON object, separated by new lines. This format is particularly useful for handling large datasets or streams of data because it allows for reading, writing, and processing one line (or one JSON object) at a time, rather than needing to load an entire file into memory at once.
# 
# The text is then split into smaller chunks, and embedded into vector format.

# %%
# loader = TextLoader("../data/state_of_the_union.txt", encoding='utf-8')
# Windows only install: 
# ! pip install https://jeffreyknockel.com/jq/jq-1.4.0-cp311-cp311-win_amd64.whl
# Other platforms
# ! pip install jq
#

loader = JSONLoader(
    file_path='./data/healthcare/augmented_notes_1000.jsonl',
    jq_schema='.note',
    json_lines=True # TODO: tell audience what json lines are
)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
# embeddings = FastEmbedEmbeddings()

# %% [markdown]
# Run the following two blocks to create and print the connection string that will be used to connect to InterSystems IRIS. 

# %%
username = 'demo'
password = 'demo' 
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '61209' # '1972'
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

# %%
# print(os.environ.get("OPENAI_API_KEY"))
# print(CONNECTION_STRING)


# %% [markdown]
# 
# The following code block will initialize a database in InterSystems IRIS, which you will later populate with the text documents that we have processed and embedded. 
# 
# This setup is essential for applications involving search and retrieval of information where the semantic content of the documents is more important than their keyword content. The vector database uses embeddings to perform similarity searches, offering significant advantages over traditional search methods by understanding the context and meaning embedded within the text. 

# %%
COLLECTION_NAME = "augmented_notes"

db = IRISVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

# %%
# If reconnecting to the database, use this:

# db = IRISVector(
#     embedding_function=embeddings,
#     dimension=1536,
#     collection_name=COLLECTION_NAME,
#     connection_string=CONNECTION_STRING,
# )

# %% [markdown]
# Run the following code block to add the documents to the newly initialized database. 

# %%
# To add documents to existing vector store:

# db.add_documents(documents)

# %% [markdown]
# Confirm that there are 1,000 documents in your vector storage by running the following block.

# %%
# print(f"Number of docs in vector store: {len(db.get()['ids'])}")

# %% [markdown]
# ## Try out vector search 
# 
# Now that the text documents are loaded, embedded, and stored in the vector database, you can try running a vector search. In the code block below, set `query` equal to "19 year old patient" and run the block. 
# 
# The second line in the block returns the documents along with their similarity scores, which quantify how similar each document is to the query. Lower scores indicate greater relevance.

# %%
query = "19 year old patient with drugs"
docs_with_score = db.similarity_search_with_score(query)

# %% [markdown]
# Run the following block to print the returned documents along with their scores.

# %%
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)

# %% [markdown]
# In the following two blocks, you will add a new document to the database and perform a similarity search on the contents of this document. Set the `content` variable to a word or phrase, then run the block. 
# 
# Printing the first returned document in the list shows that the document itself is returned as the most similar, with a similarity score of 0.0. 
# 
# Run the following block to see what else was returned by the similarity search.

# %%
content=""
db.add_documents([Document(page_content=content)])
docs_with_score = db.similarity_search_with_score(content)
docs_with_score[0]

# %%
docs_with_score

# %%
retriever = db.as_retriever()
print(retriever)


