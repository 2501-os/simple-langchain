from dotenv import load_dotenv

load_dotenv()

from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

question = "What are the rules regarding ..."

docs = vectordb.similarity_search(question, k=4)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

retriever=vectordb.as_retriever()

# any place for initial prompt?
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
)

question = "What are the rules regarding ..."
result = qa({"question": question})
print(result['answer'])

question = "What rules change when it is ..."
result = qa({"question": question})
print(result['answer'])
