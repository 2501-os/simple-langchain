from dotenv import load_dotenv

load_dotenv()

from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
print(vectordb._collection.count())


# maxmimum marginal relevance MMR, enforces diversity in search results
question = "what are the rules for CVCs for MSBs?"

result = vectordb.max_marginal_relevance_search(question, k=4, fetch_k=5)
for i, r in enumerate(result):
    print(i, r.page_content, '\n')

# self query retriever to filter for possible metadata
metadata_field_info = [
    # for multiple documents. can specify the source that the retrieved chunks come from
    # AttributeInfo(
    #     name="source",
    #     description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
    #     type="string",
    # ),
    AttributeInfo(
        name="page",
        description="The page from the pdf",
        type="integer",
    ),
]

document_content_description = "guidance pdf"
llm = OpenAI(temperature=0.9)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True,
)

question = "what did they say about BSA regulations on the 5th page?"

# this accurately only pulls chunks from the 5th page only
result = retriever.get_relevant_documents(question)
for r in result:
    print(r.metadata)

# other methods: 
# compression - summarizing returned chunks with llm
# SVM based retrieval (instead of vectordb)
# TF-IDF based retrieval (instead of vectordb)
