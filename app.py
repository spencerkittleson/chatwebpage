#!/bin/env python3
import streamlit as st
from streamlit_chat import message
from rag import ChatWebPage
from dotenv import dotenv_values

st.set_page_config(page_title="ChatWebPage")

def display_messages():
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_ingest_page():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    webpage = st.session_state["webpage_input"]
    with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {webpage}"):
        st.session_state["assistant"].ingest(webpage)


def page():

    if len(st.session_state) == 0:
        config = dotenv_values(".env")
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatWebPage(
            config['model'], config['ollama_url'])

    st.header("Chat with a web page")
    st.text_input("Url", key="webpage_input",
                  on_change=read_and_ingest_page)
    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == '__main__':
    page()