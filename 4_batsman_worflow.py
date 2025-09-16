from langgraph.graph import StateGraph, START,END
from typing import TypedDict

class BatsmanState(TypedDict):
    runs:int
    balls:int
    fours:int
    sixes:int

    sr:float
    bpb:float
    boundary_percent:float
    summary_str:str

def calculate_sr(state:BatsmanState)->BatsmanState:
    sr=(state['runs']/state['balls'])*100

    return {'sr':sr}

def calculate_bpb(state: BatsmanState):

    bpb = state['balls']/(state['fours'] + state['sixes'])

    return {'bpb': bpb}

def calculate_boundary_percent(state:BatsmanState)->BatsmanState:
    state['boundary_percent']= (((state['fours']*4)+(state['sixes']*6))/state['runs'])*100
    return state

def summary(state: BatsmanState):

    state['summary'] = f"""
Strike Rate - {state['sr']} \n
Balls per boundary - {state['bpb']} \n
Boundary percent - {state['boundary_percent']}
"""
    
    return state

graph=StateGraph(BatsmanState)

graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('summary', summary)


graph.add_edge(START,'calculate_sr')
graph.add_edge(START,'calculate_bpb')
graph.add_edge(START,'calculate_boundary_percent')
graph.add_edge('calculate_sr','summary')
graph.add_edge('calculate_bpb','summary')
graph.add_edge('calculate_boundary_percent','summary')
graph.add_edge('summary',END)

workflow=graph.compile()

initial_state = {
    'runs': 120,
    'balls': 50,
    'fours': 8,
    'sixes': 4
}
res=workflow.invoke(initial_state)

print(res)