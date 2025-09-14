from langgraph.graph import StateGraph, START,END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model='gemini-2.0-flash'
)

class BlogState(TypedDict):
    title:str
    outline:str
    blog:str

def create_outline(state:BlogState)->BlogState:
    title=state['title']
    prompt=f"Generate a detailed outline for a blog on the topic {title}"
    state['outline']=model.invoke(prompt).content

    return state

def create_blog(state:BlogState)->BlogState:
    title=state['title']
    outline=state['outline']

    prompt=f"Write a detailed blog on the title {title} using the following outline \n {outline}"
    state['blog']=model.invoke(prompt).content
    return state

graph=StateGraph(BlogState)

graph.add_node('create_outline',create_outline)
graph.add_node('create_blog',create_blog)

graph.add_edge(START,'create_outline')
graph.add_edge('create_outline','create_blog')
graph.add_edge('create_blog',END)

workflow=graph.compile()

initial_state={'title':'Use of ai in modern healthcare','outline':'','blog':''}
final_res=workflow.invoke(initial_state)
print(final_res['outline'])

print(final_res['blog'])

