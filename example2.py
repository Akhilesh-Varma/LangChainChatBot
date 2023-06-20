## Integrate code with OpenAI API 

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain

from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

# Streamlit framework
st.title("LinkedIn Data Science Job scan")

input_text = st.text_input("Search for any job related information ")

# Prompt template
first_import_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about job {name}"
)

# Memory

job_name_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
origin_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
skills_memory = ConversationBufferMemory(input_key='name', memory_key='description_history')


# OpenAI LLM

llm = OpenAI(temperature = 0.8)

chain = LLMChain(llm = llm, prompt = first_import_prompt, verbose = True, output_key = 'Job', memory  = job_name_memory)

# Prompt template
second_import_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "When did {name} start ?"
)

chain2 = LLMChain(llm = llm, prompt = second_import_prompt, verbose = True, output_key = 'date', memory = origin_memory)

# Prompt template
third_import_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "What are the skills required in {name} ?"
)

chain3 = LLMChain(llm = llm, prompt = third_import_prompt, verbose = True, output_key = 'skills', memory = skills_memory)


parent_chain = SequentialChain(chains= [chain, chain2, chain3],input_variables = ['name'], output_variables = ['Job', 'date','skills'], verbose = True)


if input_text:
    st.write(parent_chain({'name' :input_text}))

    with st.expander('Job Name'): 
        st.info(job_name_memory.buffer)

    with st.expander('Origin Events'): 
        st.info(origin_memory.buffer)

    with st.expander('Major Skills'): 
        st.info(skills_memory.buffer)