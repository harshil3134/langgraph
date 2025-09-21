from langgraph.graph import StateGraph, START,END
from dotenv import load_dotenv
from typing import TypedDict, Literal, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
import operator
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

gen_model=ChatGoogleGenerativeAI(model='gemini-2.5-pro')
eval_model=ChatGoogleGenerativeAI(model='gemini-2.0-flash')
opt_model=ChatGoogleGenerativeAI(model='gemini-2.5-pro')

class TweetEvaluation(BaseModel):
    evaluation:Literal['approved','improvement_req']=Field(...,description="final evaluation result")
    feedback:str=Field(...,description="feedback for the tweet")

structured_evaluator=eval_model.with_structured_output(TweetEvaluation)

class TweetState(TypedDict):
    topic:str
    tweet:str
    evaluation:Literal['approved','improvement_req']
    feedback:str
    max_iteration:int
    iteration:int
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]

def generate_tweet(state:TweetState):
    topic=state['topic']
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
        Write a short, original, and hilarious tweet on the topic: {topic}.

        Rules:
        - Do NOT use question-answer format.
        - Max 280 characters.
        - Use observational humor, irony, sarcasm, or cultural references.
        - Think in meme logic, punchlines, or relatable takes.
        - Use simple, day to day english
""")
    ]
 
    response=gen_model.invoke(messages).content
    return{"tweet":response,"tweet_history":[response]}

def evaluation_tweet(state:TweetState):
    # prompt
    messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"
Iteration: "{state['iteration']}"
reject atleast for first two iteration

Use the criteria below to evaluate the tweet:

1. Originality – Is this fresh, or have you seen it a hundred times before?  
2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
3. Punchiness – Is it short, sharp, and scroll-stopping?  
4. Virality Potential – Would people retweet or share it?  
5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "improvement_req"  
- feedback: One paragraph explaining the strengths and weaknesses 
""")
]
    
    response=structured_evaluator.invoke(messages)
    return {'evaluation':response.evaluation, 'feedback': response.feedback, 'feedback_history': [response.feedback]}


def optimize_tweet(state: TweetState):

    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]

    response = opt_model.invoke(messages).content
    iteration = state['iteration'] + 1

    return {'tweet': response, 'iteration': iteration, 'tweet_history': [response]}

def route_evaluation(state: TweetState):

    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'improvement_req'
    
graph=StateGraph(TweetState)

graph.add_node('generate',generate_tweet)
graph.add_node('evaluation',evaluation_tweet)
graph.add_node('optimize',optimize_tweet)

graph.add_edge(START,'generate')
graph.add_edge('generate','evaluation')
graph.add_conditional_edges('evaluation',route_evaluation,{'approved':END,'improvement_req':'optimize'})
graph.add_edge('optimize','evaluation')

workflow = graph.compile()

print(workflow)

initial_state = {
    "topic": "java script",
    "iteration": 1,
    "max_iteration": 5
}
result = workflow.invoke(initial_state)

print(result['iteration'])
print(result['tweet_history'])