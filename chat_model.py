from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(model="gpt-3.5-turbo",temperature=2)
prediction_msg = chat.predict_messages([HumanMessage(content="what are you?")])
print(prediction_msg)
