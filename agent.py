from dotenv import load_dotenv

load_dotenv()

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.9)

llm = OpenAI(temperature=0.9)
tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("What are the current 10 largest companies by revenue in the world? In your final answer include the company name, industry, and revenue per year for each company.")

# Final Answer: According to companiesmarketcap.com, the current top 20 largest companies by revenue in the world are:
# 1. Walmart - Retail - $559 billion
# 2. Amazon - Retail - $386 billion
# 3. Sinopec Group - Oil & Gas Operations - $383 billion
# 4. State Grid - Electric Utilities - $383 billion
# 5. China National Petroleum - Oil & Gas Operations - $379 billion
# 6. Royal Dutch Shell - Oil & Gas Operations - $352 billion
# 7. Saudi Aramco - Oil & Gas Operations - $330 billion
# 8. Volkswagen Group - Auto Manufacturers - $315 billion
# 9. BP - Oil & Gas Operations - $311 billion
# 10. Apple - Technology Hardware, Storage & Peripherals - $294 billion
# 11. Exxon Mobil - Oil & Gas Operations - $265 billion
# 12. Toyota Motor - Auto Manufacturers - $264 billion
# 13. Berkshire Hathaway - Insurance: Property and Casualty - $254 billion
# 14. UnitedHealth Group - Health Care: Insurance and Managed Care - $242 billion
# 15. Samsung Electronics - Technology Hardware, Storage & Peripherals - $226 billion
# 16. SoftBank Group - Diversified Telecommunication Services - $221 billion
# 17. Glencore - Metals & Mining - $215 billion
# 18. AT&T - Diversified Telecommunication Services - $197 billion
# 19. McKesson - Health Care: Pharmacy and Other Services - $195 billion
# 20. CVS Health - Health Care: Pharmacy and Other Services - $195 billion
#
