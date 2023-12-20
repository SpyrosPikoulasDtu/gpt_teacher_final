import streamlit as st
import os
import configparser
import re
import pickle

tutor_system_message = """
Using the provided lecture lecture with timestamps, act as a tutor to answer the student's questions about {lecture}. 
The format of the timestamps in the transcript is 'From HH:MM:SS to HH:MM:SS'.

When answering questions, follow these steps:

First, consult the lecture to see if the answer is covered in the provided material.
If the answer is in the lecture, provide a response with a reference to the relevant timestamps.
If the lecture does not contain information on the question, acknowledge this by stating, 'There is no information on this topic in the lecture.'
Regardless of whether the lecture contains the answer, also provide the best possible answer based on your knowledge, but clearly distinguish this from the lecture content.
Here is the lecture:\n
{transcript}
"""


# Data Initialization
@st.cache_resource
def initialization() -> dict:
    # Lecture Details
    confParser = configparser.ConfigParser()
    confParser.read("app_options.conf")
    lectures = confParser["App"]["lectures"].split(", ")
    lecture_titles = confParser["App"]["titles"].split(", ")

    lecture_d = {}
    for i,l in enumerate(lectures):
        lecture_d[l] = {}

        # Title
        lecture_d[l]["title"] = lecture_titles[i]

        # Transcript
        transcript_file = f"data/{l}/" + [file for file in os.listdir(f"data/{l}") if file.endswith('.txt')][0]
        with open(transcript_file, 'r') as f:
            transcript = f.read()
        lecture_d[l]["transcript"] = transcript

        # Video
        lecture_d[l]["video"] = f"data/{l}/" + [file for file in os.listdir(f"data/{l}") if file.endswith('.mp4')][0]

    # Features
    features = confParser["App"]["features"].split(", ")

    # Initialize chat history
    st.session_state.messages = {
        l:[{"role": "system", "content":tutor_system_message.format(
            lecture=lecture_d[l]["title"], transcript=lecture_d[l]["transcript"])}] 
            for l in lectures
            }


    return lecture_d, features

def chat_init(lectures: dict):
    if "messages" not in st.session_state:
        st.session_state.messages = {
            l:[{"role": "system", "content":tutor_system_message.format(
                lecture=lectures[l]["title"], transcript=lectures[l]["transcript"])}] 
                for l in lectures
                }

def summary():
    math_sections = {"Matrix Multiplication":[
    "data/lecture4/matrix_operations-1.png",
    "data/lecture4/matrix_operations-2.png",
    "data/lecture4/matrix_operations-3.png"
    ],
    "Laws of Matrix Operations":[
        "data/lecture4/matrix_operations-4.png",
        "data/lecture4/matrix_operations-5.png"
    ],
    "Block Matricies and Block Multiplication":[
        "data/lecture4/matrix_operations-5.png",
        "data/lecture4/matrix_operations-4.png"
    ]
    }

    with open('notes_data2.pkl', 'rb') as file:
        contents_asnwer = pickle.load(file)

    for ms in math_sections.keys():
        with st.expander(ms):         
            clist =  contents_asnwer[ms]
            for c in clist:
                content = c[0]
                text = c[1]
                if content == "markdown":
                    st.markdown(text)
                else:
                    st.latex(text)