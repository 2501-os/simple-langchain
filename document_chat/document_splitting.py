from dotenv import load_dotenv

load_dotenv()

import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

chunk_size = 150 
chunk_overlap = 4

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""] # split by these in order
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator=' '
)

text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""
split = r_splitter.split_text(text)
for i, s in enumerate(split):
    print(i, s)

# split document
pdf_filepath = str(os.getenv('PDF_FILEPATH'))
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()

doc_r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""] # split by these in order
)

docs = doc_r_splitter.split_documents(pages)

print(len(docs))
print(len(pages))

print(docs[1])
