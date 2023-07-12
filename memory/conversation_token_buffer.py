from dotenv import load_dotenv

load_dotenv()

from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory

# conversation token buffer memory
# saves the previous X tokens in memory

llm = ChatOpenAI(temperature=0.9)

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)

memory.save_context({"input": "Who are you?"},
                    {"output": "I am me."})
memory.save_context({"input": "What's up?"},
                    {"output": "Not much."})
memory.load_memory_variables({})

print(memory)

conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

prediction = conversation.predict(input="What is the first thing I asked you?")
print(prediction)
