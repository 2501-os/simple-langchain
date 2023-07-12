from dotenv import load_dotenv

load_dotenv()

import os
import numpy as np
from langchain.vectorstores import Chroma # import first, or else it errors
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


# openai embedding model
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# testing embeddings
# sentence1 = "NATO allies offer security assurances for Ukraine on path to membership"
# sentence2 = "Russian spy chief says he spoke to CIA boss about 'what to do with Ukraine'"
# sentence3 = "China lashes back at NATO criticism, warns it will protect its rights"
#
# embedding1 = embedding.embed_query(sentence1)
# embedding2 = embedding.embed_query(sentence2)
# embedding3 = embedding.embed_query(sentence3)
#
# print(len(embedding1))
# print(len(embedding2))
# print(len(embedding3))
#
# print(np.dot(embedding1, embedding2))
# print(np.dot(embedding1, embedding3))
# print(np.dot(embedding2, embedding3))

# load docs
pdf_filepath = str(os.getenv('PDF_FILEPATH'))
loader = PyPDFLoader(pdf_filepath)
docs = loader.load()

# chunk docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
splits = text_splitter.split_documents(docs)

# create embeddings on chunks and store in vectordb
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
print(vectordb._collection.count())

# similarity search
question = "what is this guidance for?"

docs = vectordb.similarity_search(question, k=3)

print(docs[0])
print(docs[1])
print(docs[2])
