from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict
import time


class CrashState(TypedDict):
    input:str
    step_1:str
    step_2:str
    step_3:str

def step_1(state:CrashState):
    print("step 1 executed")
    return {"step_1":"done", "input":state['input']}

def step_2(state:CrashState):
    print("step 2 executing...")
    time.sleep(3)  # Just 3 seconds for demonstration
    print("step 2 completed")
    return {"step_2":"done"}

def step_3(state:CrashState):
    print("step 3 executed")
    return {"step_3":"completed"}

# 3. Build the graph
builder = StateGraph(CrashState)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.set_entry_point("step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

try:
    print("▶ Running graph: Please manually interrupt during Step 2...")
    graph.invoke({"input": "start"}, config={"configurable": {"thread_id": 'thread-1'}})
except KeyboardInterrupt:
    print(" Kernel manually interrupted (crash simulated).")

# 6. Re-run to show fault-tolerant resume
print("\nRe-running the graph to demonstrate fault tolerance...")
final_state = graph.invoke(None, config={"configurable": {"thread_id": 'thread-1'}})
print("\n Final State:", final_state)

list(graph.get_state_history({"configurable": {"thread_id": 'thread-1'}}))