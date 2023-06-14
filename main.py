## Integrate code with OpenAI API 

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

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

# OpenAI LLM

llm = OpenAI(temperature = 0.8)

chain = LLMChain(llm = llm, prompt = first_import_prompt, verbose = True)



if input_text:
    st.write(chain.run(input_text))

