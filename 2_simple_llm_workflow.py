from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model='gemini-2.0-flash'
)

class LLMState(TypedDict):
    question:str
    response:str

def llm_qa(state:LLMState)->LLMState:

    question=state['question']
    prompt=f"asnwer the following question{question}"
    state['response']=model.invoke(prompt).content
    return state

graph=StateGraph(LLMState)

graph.add_node('llm_qa',llm_qa)

graph.add_edge(START,'llm_qa')
graph.add_edge('llm_qa',END)

workflow=graph.compile()

initial_state={'question':'how much far is moon from earth?','response':''}

final_state=workflow.invoke(initial_state)

print(final_state['response'])

print(model.invoke('How far is moon from the earth?').content)
