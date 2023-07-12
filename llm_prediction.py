from dotenv import load_dotenv

load_dotenv()

from langchain.llms import OpenAI

llm = OpenAI(temperature=2)

prediction = llm.predict("what are you? describe it esoterically.")
print(prediction)
