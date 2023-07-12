from dotenv import load_dotenv

load_dotenv()

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# conversation buffer memory
# simply prepends previous inputs and outputs to a new input

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "The following is an conversation between a human and an AI. The AI is talkative and "
        "provides lots of specific details from its context."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0.9)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

result1 = conversation.predict(input="Hello.")
print(result1)

result2 = conversation.predict(input="What are you?")
print(result2)

result3 = conversation.predict(input="What was the first thing I said to you?")
print(result3)
