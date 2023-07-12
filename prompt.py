from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

template = "You are a helpful assistant that writes {programming_language} code."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

messages = chat_prompt.format_messages(programming_language="Fortran", text="Write a binary search program.")

chat = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.9,max_tokens=1000)
prediction = chat.predict_messages(messages)
print(prediction)

