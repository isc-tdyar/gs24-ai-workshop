import iris
# TODO make these searches generic per table
class VectorSearch:
    def __init__(self) -> None:
        self.conn = iris.connect('localhost',51729,'USER','demo', 'demo')
        
    def search_vector_db_with_embedding(self, query_embedding, top_k:int) -> list:
        query = f"""SELECT TOP 4 data.id
                    FROM augmented_notes data
                    ORDER BY VECTOR_DOT_PRODUCT(TO_VECTOR(data.embedding), TO_VECTOR(?)) DESC
                    """
        iris_cursor = self.conn.cursor()
        iris_cursor.execute(query, [str(query_embedding)])
        origin_list = iris_cursor.fetchall()
        return origin_list
    
    def search_q_and_a_docs(self, story_ids: list[str]) -> list:
        id_tuple = tuple(story_ids)
        print(id_tuple)
        query = f"""SELECT TOP 20 
                    FROM augmented_note
                    WHERE id IN {id_tuple}
                    """
        iris_cursor = self.conn.cursor()
        iris_cursor.execute(query)
        resultset = list(iris_cursor.fetchall())
        q_and_a_list = [{'question':q_and_a[0], 'answer':q_and_a[1]} for q_and_a in resultset]
        return q_and_a_list
import streamlit as st
# from streamlit_jupyter import StreamlitPatcher, tqdm

# StreamlitPatcher().jupyter() 

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.callbacks import get_openai_callback
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain_iris import IRISVector
import os
# from sentence_transformers import SentenceTransformer
# from vector_search import VectorSearch

from dotenv import load_dotenv
load_dotenv(override=True)

username = 'demo'
password = 'demo' 
hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
port = '51729' # '1972'
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

embeddings = OpenAIEmbeddings()

HC_COLLECTION_NAME = "augmented_notes"
db = IRISVector(
    embedding_function=embeddings,
    dimension=1536,
    collection_name=HC_COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

FINANCE_COLLECTION_NAME = "financial_tweets"
db2 = IRISVector(
    embedding_function=embeddings,
    dimension=1536,
    collection_name=FINANCE_COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot that can access your vector stores. What would you like to know?"}
    ]

st.header('GS 2024 Vector Search')


with st.sidebar:
    st.header('Settings')
    choose_embed = st.radio("Choose an embedding model (don't change for exercise):",("all-MiniLM-L6-v2","OPEN AI Embeddings","None"),index=1)
    choose_dataset = st.radio("Choose an IRIS collection:",("healthcare","finance","None"),index=2)
    choose_LM = st.radio("Choose a language model:",("gpt-3.5-turbo","gpt-4-turbo","None"),index=0)


for msg in st.session_state['messages']:
    if msg["role"] == "assistant":
        st.chat_message(msg["role"]).write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"].replace("$", "\$"))

if prompt := st.chat_input():

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt.replace("$", "\$")) # Escaping '$', otherwise Streamlit can interpret it as Latex

    llm = ChatOpenAI(
        temperature=0,
        # openai_api_key=,
        model_name=choose_LM
    )
    # Create chain. We are using Summary Memory for fewer tokens.
    conversation_sum = ConversationChain(
        llm=llm,
        memory=ConversationSummaryMemory(llm=llm),
        verbose=True
    )


    with st.chat_message("assistant"):
        #;
        # Encode the user's prompt and find the top-k similar questions in the vector DB.
        # embedding = model.encode(prompt)
        # documents = peristent_DB.search_vector_db_with_embedding(str(embedding.tolist()), top_k=4)
        # doc_content_list, doc_id_list = map(list, zip(*documents))
        # doc_list = [Document(page_content=doc_content, metadata={"source": "local"}) for doc_content in doc_content_list]
        #;
        # This can potentially return many large documents, so we should use LangChain to chunk the results:
        # text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=0)
        # docs = text_splitter.split_documents(doc_list)
        #;
        # q_and_a_docs = peristent_DB.search_q_and_a_docs(doc_id_list)

        # relevant_docs = [str(doc.page_content)[:250] for doc in docs]
        query = prompt
        docs_with_score = None
        if choose_dataset == "healthcare":
            docs_with_score = db.similarity_search_with_score(query)
        elif choose_dataset == "finance":
            docs_with_score = db2.similarity_search_with_score(query)
        else:
            print("No Dataset selected")
        # relevant_docs[:]

        template = f"""
                    Prompt: {prompt}

                    Relevant Documents: {str(docs_with_score)}

                    You should only make use of the provided Relevant Documents. They are important information belonging to the user, and it is important that any advice you give is grounded in these documents. If the documents are irrelevant to the question, simply state that you do not have the relevant information available in the database.
                """
        resp = conversation_sum(template)
        
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Hi, I'm a chatbot that can access your vector stores. What would you like to know?"}
            ]

        st.session_state.messages.append({"role": "assistant", "content": resp['response']})
        st.write(resp['response'].replace("$", "\$"))