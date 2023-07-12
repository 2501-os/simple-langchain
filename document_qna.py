from dotenv import load_dotenv

load_dotenv()

import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

# DocArrayInMemorySearch stores embeddings in memory

csv_filepath = str(os.getenv('CSV_FILEPATH'))
loader = CSVLoader(file_path=csv_filepath)

# create an embeddings vector store (db) from csv file
# document is chunked, embeddings are created from chunks, then embedding vectors are saved in db along with chunks
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

# embedding is created for query, then embedding is compared to all vectors in db, similar ones are returned
query = "Describe all of the columns of this dataset and what they represent in a markdown table."
response = index.query(query)
print(response)

# --- doing it manually ---

docs = loader.load()

embeddings = OpenAIEmbeddings()

# create embeddings on file and store embedding vectors in memory
db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)

# run a numerical embedding similary search to match the query to the vector embeddings
query2 = "List X ..."
result_docs = db.similarity_search(query2)
print(len(result_docs))

# ----

# --- doing it with langchain methods ---

# abstract fn that retrieves vectors embeddings in a store from some query based on some specified method
retriever = db.as_retriever()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

# 1. does retrieval via embeddings similarity search
# 2. then does question answering over the retrieved documents via llm
# different interesting chain_types: map_reduce, refine, map_rerank
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", # stuff all retrieved documents into llm, then respond to the prompt
    retriever=retriever, 
    verbose=True
)

query3 = "List X ..."
response = qa_stuff.run(query3)
print(response)
