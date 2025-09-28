from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class JokeState(TypedDict):
    topic:str
    joke:str
    explanation:str

def generate_joke(state:JokeState):
    prompt=f"generate a joke on the topic ${state['topic']}"
    response=model.invoke(prompt).content

    return {'joke':response}

def generate_explanation(state:JokeState):
    prompt=f'write an explanation for the joke {state['joke']}'
    response=model.invoke(prompt).content

    return {'explanation':response}

graph=StateGraph(JokeState)

graph.add_node('generate_joke',generate_joke)
graph.add_node('generate_explanation',generate_explanation)

graph.add_edge(START,'generate_joke')
graph.add_edge('generate_joke','generate_explanation')
graph.add_edge('generate_explanation',END)

checkpointer=InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

config1 = {"configurable": {"thread_id": "1"}}
# res=workflow.invoke({'topic':'pizza'}, config=config1)


# print(res)

# print("------",workflow.get_state(config1))

# config2 = {"configurable": {"thread_id": "2"}}
# res2=workflow.invoke({'topic':'pasta'}, config=config2)

# print(res2)

# print("------",workflow.get_state(config2))

#Time travel

print(list(workflow.get_state_history(config1)))

workflow.get_state({"configurable":{"thread_id":"1","checkpoint_id":"1f09c62a-1e44-623c-bfff-c236cae258cb"}})

print('------',workflow.invoke(None, {"configurable":{"thread_id":"1","checkpoint_id":"1f09c62a-1e44-623c-bfff-c236cae258cb"}}))