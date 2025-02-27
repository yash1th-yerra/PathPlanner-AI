import streamlit as st
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.schema.runnable import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

os.environ["GEMINI_API_KEY"] = "AIzaSyDsFM65POh3Nfq2vfLMVi_INEaqHNrKtXY"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")


prompt_template = PromptTemplate(
    template="""
    You are an AI travel assistant. A user wants to travel from {source} to {destination}. 
    Provide travel options for cab, train, bus, and flight with estimated prices and travel times.
    """
)



chain = {"source": RunnablePassthrough(), 
         "destination": RunnablePassthrough(), 
         "preference": RunnablePassthrough()} | prompt_template | llm


st.title("AI-Powered Travel planner")


source = st.text_input("Enter Source Location")
destination = st.text_input("Enter Destination Location")


if st.button("Find Travel Options"):
    if source and destination:
        response = chain.invoke({"source":source,"destination":destination})
        st.write(response.content)
    else:
        st.warning("Please enter both source and destination")