from langgraph.graph import StateGraph, START,END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

load_dotenv()

llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

def chat_node(state:ChatState):
    messages=state['messages']

    response=llm.invoke(messages)
    return{'messages':[response]}

graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile()

initial_state = {
    'messages': [HumanMessage(content='What is the name of longest river?')]
}

print(chatbot.invoke(initial_state)['messages'][-1].content)