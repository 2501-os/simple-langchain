from dotenv import load_dotenv

load_dotenv()

from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

# conversation summary buffer
# uses an llm to summarize previous memory inputs/outputs, reducing the 
# memory's total length

llm = ChatOpenAI(temperature=0.9)

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})
memory.load_memory_variables({})

conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

print(memory)

prediction = conversation.predict(input="What would be a good demo to show?")
print(prediction)
