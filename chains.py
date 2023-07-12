from dotenv import load_dotenv

load_dotenv()

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

chat = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.9,max_tokens=1000)

# create system prompt
template = "You are a helpful assistant that writes {programming_language} code."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# create chat input
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# combine
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# call llm
# this is the same as in prompt.py
# though instead of direct .predict_messages an LLMChain class is used
chain = LLMChain(llm=chat, prompt=chat_prompt)
prediction = chain.run(programming_language="Fortran", text="Write a binary search program.")
print(prediction)
