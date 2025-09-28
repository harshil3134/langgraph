import streamlit as st
from langchain_core.messages import HumanMessage
from langraph_backend import chatbot

if 'message_history' not in st.session_state:
    st.session_state.message_history=[]
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

user_input=st.chat_input('Type here')
message_history=[]
for i in st.session_state['message_history']:
    if i['role']=='user':
         with st.chat_message('user'):
            st.text(i['content'])
    if i['role']=='assistant':
         with st.chat_message('assistant'):
            st.text(i['content'])



if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    response=chatbot.invoke({'messages':[HumanMessage(content=user_input)]},config=CONFIG)
    ai_message = response['messages'][-1].content

    st.session_state['message_history'].append({'role':'assistant','content':ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)


