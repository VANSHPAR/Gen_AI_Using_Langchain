import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

#page config
st.set_page_config(page_title="Simple Langchain Chatbot with Groq",page_icon="🤖")
#title
st.title("🤖 Simple Langchain Chat with Groq")
st.markdown("Learn Langchain basics with Groq's ultra fast Inference!")

with st.sidebar:
    st.header("Settings")

    #API key
    api_key=st.text_input("Groq API Key",type="password",help="Get Free API Key at Groq.com")

    #model seleection
    model_name=st.selectbox(
        "Model",
        ["llama-3.1-8b-instant","openai/gpt-oss-120b","qwen/qwen3-32b"],
        index=0
    )

    #clear button
    if st.button("Clear Chat"):
        st.session_state.messages=[]
        st.rerun()

#intilize chat history
if "messages" not in st.session_state:
    st.session_state.messages=[]

#intilize llm 
@st.cache_resource
def get_chain(api_key,model_name):
    if not api_key:
        return None
    
    #intilize groq model

    llm=ChatGroq(groq_api_key=api_key,
             model_name=model_name,
             temperature=0.7,
             streaming=True)
    
    prompt=ChatPromptTemplate([
        ("system","You are a helpful assistant powered by Groq. Answer questions clearly and concisely."),
        ("user","{question}")
    ])

    #create chain
    chain=prompt | llm | StrOutputParser()

    return chain

chain=get_chain(api_key,model_name)

if not chain:
    st.warning("Please enter your Groq API key in the sidebar")
    st.markdown("[Get your free API key here](https://console.groq.com/keys)")

else:
    #display
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    #chat input
    if question:=st.chat_input("Ask me anything"):
        #add user message to session state
        st.session_state.messages.append({"role":"user","content":question})
        with st.chat_message("user"):
            st.write(question)

        #Generate Response
        with st.chat_message("assistant"):
            message_placeholder=st.empty()
            full_response=""

            try:
                #response from groq
                for chunk in chain.stream({"question":question}):
                    full_response+=chunk
                    message_placeholder.markdown(full_response + " ")
                message_placeholder.markdown(full_response)

                #add to history
                st.session_state.messages.append({"role":"assistant","content":full_response})

            except Exception as e:
                st.error(f"Error :{str(e)}")

st.markdown("---")
st.markdown("### 💡Try these examples:")
col1,col2=st.columns(2)

with col1:
    st.markdown("- What is Langchain?")
    st.markdown("- Explain Groq's LPU Technology")

with col2:
    st.markdown("- How do i learn python programming??")
    st.markdown("- Write a haiku about AI")

#footer
st.markdown("---")
st.markdown("Built with Langchain & Groq | Experience the speed! ⚡")
