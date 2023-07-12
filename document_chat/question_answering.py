from dotenv import load_dotenv

load_dotenv()

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

print(vectordb._collection.count())

question = "..."

# get document chunks from vector db
docs = vectordb.similarity_search(question,k=3)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# optional template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# feed document chunks along with question to llm for an answer
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    chain_type="stuff" # map_reduce, refine, ..
)

result = qa_chain({"query": question})
print(result["result"])
