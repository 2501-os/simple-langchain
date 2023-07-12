from dotenv import load_dotenv

load_dotenv()

from langchain import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI

# conversation buffer window memory
# saves K previous input/output pairs

memory = ConversationBufferWindowMemory(k=1)

# manually creating a previous memory (just setting inputs and outputs)
memory.save_context({"input": "Who are you?"},
                    {"output": "I am me."})
memory.save_context({"input": "What's up?"},
                    {"output": "Not much."})
memory.load_memory_variables({})

llm = ChatOpenAI(temperature=0.9)
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

prediction1 = conversation.predict(input="What is the first thing I asked you?")
print(prediction1)

prediction2 = conversation.predict(input="What is the first thing I asked you?")
print(prediction2)
