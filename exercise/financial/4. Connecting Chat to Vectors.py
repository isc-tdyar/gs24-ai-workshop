       
# Import the Python libaries that will be used for this app.
# Libraries of note: 
# Streamlit, a Python library that makes it easy to create and share beautiful, custom web apps for data science and machine learning.
# ChatOpenAI, a class that provides a simple interface to interact with OpenAI's models.
# ConversationChain and ConversationSummaryMemory, classes that represents a conversation between a user and an AI and retain the context of a conversation.
# OpenAIEmbeddings, a class that provides a way to perform vector embeddings using OpenAI's embeddings.
# IRISVector, a class that provides a way to interact with the IRIS vector store.
import streamlit as st
from langchain_community.chat_models import ChatOpenAI 
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_iris import IRISVector
import os

# Import dotenv, a module that provides a way to read environment variable files, and load the dotenv (.env) file that provides a few variables we need
from dotenv import load_dotenv
load_dotenv(override=True)

# Load the urlextractor, a module that extracts URLs and will enable us to follow web-links
from urlextract import URLExtract
extractor = URLExtract()

# Define the IRIS connection - the username, password, hostname, port, and namespace for the IRIS connection.
username = 'demo'  # This is the username for the IRIS connection
password = 'demo'  # This is the password for the IRIS connection
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')  
port = '61209'  # This is the port number for the IRIS connection
namespace = 'USER'  # This is the namespace for the IRIS connection

# Create the connection string for the IRIS connection
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

# Create an instance of OpenAIEmbeddings, a class that provides a way to perform vector embeddings using OpenAI's embeddings.
embeddings = OpenAIEmbeddings()

# *** Instantiate IRISVector ***
##TODO: Remove code from this line through line 72.
# Define the name of the healthcare collection in the IRIS vector store.
# HC_COLLECTION_NAME = "augmented_notes"

# # Create an instance of IRISVector, which is a class that provides a way to interact with the IRIS vector store.
# # This instance is for the healthcare collection, and it uses the OpenAI embeddings.
# # The dimension of the embeddings is set to 1536, and the collection name and connection string are specified.
# db = IRISVector(
#     # The embedding function to use for the vector embeddings.
#     embedding_function=embeddings,
#     # The dimension of the embeddings (in this case, 1536).
#     dimension=1536,
#     # The name of the collection in the IRIS vector store.
#     collection_name=HC_COLLECTION_NAME,
#     # The connection string to use for connecting to the IRIS vector store.
#     connection_string=CONNECTION_STRING,
# )

# Define the name of the finance collection in the IRIS vector store.
FINANCE_COLLECTION_NAME = "financial_tweets"

# Create another instance of IRISVector, this time for the finance collection.
# It also uses the OpenAI embeddings, and has the same dimension and connection string as the healthcare collection.
db2 = IRISVector(
    # The embedding function to use for the vector embeddings.
    embedding_function=embeddings,
    # The dimension of the embeddings (in this case, 1536). This is 1536 because OpenAI Embeddings use that size
    dimension=1536,
    # The name of the collection in the IRIS vector store.
    collection_name=FINANCE_COLLECTION_NAME,
    # The connection string to use for connecting to the IRIS vector store.
    connection_string=CONNECTION_STRING,
)

### Used to have a starting message in our application
# Check if the "messages" key exists in the Streamlit session state.
# If it doesn't exist, create a new list and assign it to the "messages" key.
if "messages" not in st.session_state:
    # Initialize the "messages" list with a welcome message from the assistant.
    st.session_state["messages"] = [
        # The role of this message is "assistant", and the content is a welcome message.
        {"role": "assistant", "content": "Hi, I'm a chatbot that can access your vector stores. What would you like to know?"}
    ]

# *** Add a title for the application ***
# TODO: delete the existing title
# This line creates a header in the Streamlit application with the title "GS 2024 Vector Search"
st.header('GS 2024 Vector Search')

# *** Customize the UI ***
# In streamlit we can add settings using the st.sidebar
with st.sidebar:
    st.header('Settings')
    # 1. A selection for our embedding model
    # choose_embed = st.radio("Choose an embedding model (don't change for exercise):",("OpenAI Embedding","None"),index=0)
    # 2. We let our users select what vector store to query against
    # choose_dataset = st.radio("Choose an IRIS collection:",("healthcare","finance"),index=1)
    # 3. We let our uses choose which AI model we want to power our chatbot
    choose_LM = st.radio("Choose a language model:",("gpt-3.5-turbo","gpt-4-turbo"),index=0)
    # 4. If the user selected financial dataset, ask if they want to preprocess information
    explain = st.radio("Show explanation?:",("Yes", "No"),index=0)
    # link_retrieval = st.radio("Retrieve Links?:",("No","Yes"),index=0)

# In streamlet, we can add our messages to the user screen by listening to our session
for msg in st.session_state['messages']:
    # If the "chat" is coming from AI, we write the content with the ISC logo
    if msg["role"] == "assistant":
        st.chat_message(msg["role"]).write(msg["content"])
    # If the "chat" is the user, we write the content as the user image, and replace some strings the UI doesn't like
    else:
        st.chat_message(msg["role"]).write(msg["content"].replace("$", "\$"))

# Check if the user has entered a prompt (input) in the chat window
if prompt := st.chat_input(): 

    # Add the user's input to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display the user's input in the chat window, escaping any '$' characters
    st.chat_message("user").write(prompt.replace("$", "\$"))

# *** Change the precision of the AI model ***
# Update the temperature to a value between 0 and 1 to adjust the precision - 0 is the most precise
    # Create an instance of the ChatOpenAI class, which is a language model
    llm = ChatOpenAI(
        temperature=0,  # Set the temperature for the language model (0 is default)
        model_name=choose_LM  # Use the selected language model (gpt-3.5-turbo or gpt-4-turbo)
    )
 # *** Create a ConversationChain Instance ***
 # This uses the language model (llm) and a ConversationSummaryMemory instance for summarizing the conversation
    conversation_sum = ConversationChain(
        llm=llm,  # The language model to use
        memory=ConversationSummaryMemory(llm=llm),  # Summarize the conversation
        verbose=True  # Set verbosity to True (optional)
    )

    # Here we respond to the user based on the messages they receive 
    with st.chat_message("assistant"):
        # We rename our prompt (user input) to query to better illustrate that we'll compare it to the vector store
        query = prompt
        # We'll store the most similar results from the vector database here
        docs_with_score = None
        # Based on the dataset, we will compare the user query to the proper vector store
        # if choose_dataset == "healthcare":
        #     # If Healthcare, that's db (collection name HC_COLLECTION_NAME)
        #     docs_with_score = db.similarity_search_with_score(query)
        # elif choose_dataset == "finance":
            # If Finance, that's db2 (collection name FINANCE_COLLECTION)
        docs_with_score = db2.similarity_search_with_score(query)
        # else:
        #     # If Nothing, we have No Context
        #     print("No Dataset selected")
        print(docs_with_score)
        # Here we build the prompt for the AI: Prompt is the user input and docs_with_score is the vector database result
        relevant_docs = ["".join(str(doc.page_content)) + " " for doc, _ in docs_with_score]
        # if link retrieval, then try to scrape the content from the page 
        
        # Prefetch the first returned link and include it in the documents
        
        # if link_retrieval == "Yes":
        #     first_relevant_doc = relevant_docs
        #     urls = extractor.find_urls(str(first_relevant_doc))
        #     print(urls) # prints: ['stackoverflow.com']     
        #     web_loader = SeleniumURLLoader(urls[:1])
        #     web_docs = web_loader.load()
        #     print(web_docs)
        #     pass
        
    # *** Create LLM Prompt ***
    ##TODO: Remove code from this line through line 171. 
        template = f"""
Prompt: {prompt}

Relevant Documents: {relevant_docs}

You should only make use of the provided Relevant Documents. They are important information belonging to the user, and it is important that any advice you give is grounded in these documents. If the documents are irrelevant to the question, simply state that you do not have the relevant information available in the database.
                """
        # And our response is taken care of by the conversation summarization chain with our template prompt
        # chunks = []
        # for chunk in conversation_sum.stream(template):
        #     chunks.append(chunk)
        # print(chunks)
        resp = conversation_sum(template)
        
        # Finally, we make sure that if the user didn't put anything or cleared session, we reset the page
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hi, I'm a chatbot that can access your vector stores. What would you like to know?"}
            ]

        # And we add to the session state the message history
        st.session_state.messages.append({"role": "assistant", "content": resp['response']})
        print(resp)
        # And we also add the response from the AI
        st.write(resp['response'].replace("$", "\$"))
        if explain == "Yes":
            with st.expander("Supporting Evidence"):
                for doc, _ in docs_with_score[:1]:
                    doc_content = "".join(str(doc.page_content))
                    # st.write(f"""Here are the relevant documents""")
                    st.write(f"""{doc_content}""")
                    urls = extractor.find_urls(doc_content)
                    print(urls) # prints: ['stackoverflow.com']     
                    for url in urls:       
                        st.page_link(url, label="Source")