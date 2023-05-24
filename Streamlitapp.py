# Bring in Deps
import os
from apikey import openaikey

import streamlit as st
##from langchain.utilities import scholarly
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

#from langchain.utilities import scholarly #Replace with Google Scholar API Wrapper


os.environ['OPENAI_API_KEY'] = openaikey

# App Framework
st.title('🦜🔗⚛️Athena')
prompt= st.text_input('Ask a Fusion Question')


# Prompt Templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template=' {topic}')

script_template = PromptTemplate(
    input_variables = ['title'], #'scholarly_research'],
    template='suggest research papers with authors related to title TITLE:{title}') #while cross referencing with Google Scholar:{scholarly}') #replace with google scholar research
    
 # Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')



#LLMs
llm = OpenAI(temperature=0.7)

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key= 'title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key= 'script', memory=script_memory)

#scholar = scholarly()
# (Example) sequential_chain = SequentialChain(chains=[title_chain,script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)


# Prompt Screen
if prompt:
    title = title_chain.run(prompt)
    #scholarly = scholar.run(prompt) #replace with google scholar research
    script = script_chain.run(title=title) #scholarly=scholarly)
   # (Example) response = sequential_chain({'topic':prompt})
    st.write(title)
    st.write(script)

    with st.expander('Q&A'):
        st.info(title_memory.buffer)
    with st.expander('Research Citation'):
        st.info(script_memory.buffer)
        
    #with st.expander('Wikipedia Research'):
        st.info(scholarly) #replace with google scholar research