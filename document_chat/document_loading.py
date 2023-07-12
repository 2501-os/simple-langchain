from dotenv import load_dotenv

load_dotenv()

import os
from langchain.document_loaders import PyPDFLoader

pdf_filepath = str(os.getenv('PDF_FILEPATH'))
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()

print(len(pages))
print(pages[0])
