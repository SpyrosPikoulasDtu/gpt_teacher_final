import streamlit as st
import configparser
import app_modules
import json
from openai import OpenAI
import streamlit_scrollable_textbox as stx

# import dotenv
# dotenv.load_dotenv()

st.set_page_config(layout="wide")

# Data Initialization
lectures, features = app_modules.initialization()
titles_to_lectures = {lectures[l]["title"]:l for l in list(lectures.keys())}
app_modules.chat_init(lectures)

# Header
st.markdown(
        "<h1 style='text-align: center;'>Generative Pretrained Teacher 0.1</h1>",
        unsafe_allow_html=True,
)
st.markdown("<div style='padding: 20px;'></div>", unsafe_allow_html=True)

# Lecture Selection
st.sidebar.markdown("## Lecture selection")
selected_lecture_title = st.sidebar.selectbox(
    "Select a lecture:",
    list(titles_to_lectures.keys()),
)
selected_lecture = titles_to_lectures[selected_lecture_title]

# Lecture Title
st.markdown(
        f"<h3 style='text-align: center;'>{selected_lecture_title}</h3>",
        unsafe_allow_html=True,
)
st.markdown("<div style='padding: 20px;'></div>", unsafe_allow_html=True)

# Video
columns = st.columns(2)
video_file = lectures[selected_lecture]["video"]
with columns[0]:
    st.video(video_file, format="video/mp4", start_time=0)
with columns[1]:
    stx.scrollableTextbox(lectures[selected_lecture]["transcript"], height=400)
st.markdown("<div style='padding: 20px;'></div>", unsafe_allow_html=True)

# Feature Selection
st.sidebar.markdown("## Feature selection")
selected_feature = st.sidebar.selectbox(
    "Select a feature:",
    features,
)

### Tutor Feature ###
if selected_feature == "Tutor":
    # Display chat messages from history on app rerun
    for message in st.session_state.messages[selected_lecture]:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Insert here..."):
        # Add user message to chat history
        st.session_state.messages[selected_lecture].append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant with stream
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in OpenAI().chat.completions.create(
                model="gpt-4-1106-preview",
                temperature=0,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[selected_lecture]
                ],
                stream=True,
                ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assitant message to chat history
        st.session_state.messages[selected_lecture].append({"role": "assistant", "content": full_response})

if selected_feature == "Summary":
    if selected_lecture_title != "Matrices Multiplication":
        st.text("Sorry, Only Lecture4, Matrices Multiplicaiton supports Summary, since it was the only \none with with structured and content rich pdf notes(book of the lecture)")
    else:
        app_modules.summary()

### Adaptive Feature ###

class Student:
    """
    Data structure that will hold student data.
    This will be more efficient that keeping just a list of conversation pieces (Langchain Memory).
    We can then "remind" the model at each prompt what the student's level, interests etc. are.
    """
    def __init__(self):
        self.state = "Neutral"
        self.understood_topics = []
        self.learning_topics = []

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


sentiment_analysis_prompt = PromptTemplate(
        input_variables=["question"],
        output_variable=["sentiment"],
        template="""
You are a tutor. You will be given a student's question about a subject.
Based on the question, identify the student's sentiment. 
Respond with their sentiment as a single word like: Neutral, Frustrated, Happy, Sad, Dissapointed, Interested

Question: {question}
""")

modified_answer_template = PromptTemplate(
        input_variables=["sentiment", "answer"],
        output_variable=["answer"],
        template="""
This is an answer to a student's question.
Please rephrase it based on their current emotional state in order to help the student better cope with the learning process. 

The student's state: {sentiment}

The answer: {answer}
        """
)

llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

answer_chain = (
    ChatPromptTemplate.from_template("{question}")
    | llm
    | {"answer": RunnablePassthrough()}
)

student = Student()

def found_sentiment(sentiment):
    student.state = sentiment
    return sentiment

sentiment_analysis_chain = (
    sentiment_analysis_prompt 
    | llm 
    | RunnableLambda(found_sentiment)
)

modified_answer_chain = (
    modified_answer_template |
    llm
)

student_adaptation_chain = (
    {
        "sentiment": sentiment_analysis_chain,
        "answer": answer_chain
    }
    | modified_answer_chain
)

if selected_feature == "Adaptive":

    for message in st.session_state.messages[selected_lecture]:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Insert here..."):
        # Add user message to chat history
        st.session_state.messages[selected_lecture].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
                st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = student_adaptation_chain.invoke({"question": prompt})

            if hasattr(response, "content"):
                full_response = response.content
            else:
                full_response = response
            message_placeholder.markdown(full_response)

            st.session_state.messages[selected_lecture].append({"role":
                                                                "assistant",
                                                                "content":
                                                                full_response})


### Test Creation Feature ###
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from LectureIndex import LectureIndex
from LectureLoader import LectureLoader
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit_book as stb

@st.cache_data()
def get_lecture_index():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    lecture_loader = LectureLoader.from_folder("data/lecture1", add_lecture_info=True)
    lecture_docs = lecture_loader.load()
    lecture_index = LectureIndex.from_documents(lecture_docs, embeddings)

    return lecture_index

lecture_index = get_lecture_index()

response_schemas = [
    ResponseSchema(name="question", description="A multiple choice question from input text snippet."),
    ResponseSchema(name="options", description="Possible choices for the multiple choice question.", type="List[string]"),
    ResponseSchema(name="answer", description="Index of the correct answer for the question.", type="int"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions(only_json=True)


test_prompt = ChatPromptTemplate(
    messages = [
        SystemMessagePromptTemplate.from_template("""
        Given a text input, generate multiple choice questions along with the correct answer.
        
        {format_instructions}. 
        Make sure to surround each question with ```json\n and ```.
        """),
        HumanMessagePromptTemplate.from_template("""
        {user_prompt}
        Make sure the questions follow these learning objectives: {learning_objectives}
        Make three questions.
        """)
    ],
    input_variables=["user_prompt", "learning_objectives"],
    partial_variables={"format_instructions": format_instructions}
)

@st.cache_data()
def get_relevant_snippets(learning_objectives_str):
    learning_objectives = learning_objectives_str.split(",")
    text = ""
    for learning_objective in learning_objectives:
        results = lecture_index.similarity_search_with_score_threshold(learning_objective, 1.0)
        text += "".join([result.page_content for result in results])
    return text

if selected_feature == "Tests":

    st.markdown("## Test Creation")
    learning_objectives = st.text_input("Learning Objectives", value="Learning Objectives", help="Learning objectives you want the test to be based on.")

    if st.button("Create Test"):
        
        snippets = get_relevant_snippets(learning_objectives)
        user_query = test_prompt.format_prompt(user_prompt=snippets,
                                               learning_objectives=learning_objectives)
        response = llm(user_query.to_messages())

        response = response.content
        print(response)

        response = response.split('```json\n')[1:]
        response = [r.replace("```", "") for r in response]
        response = [json.loads(r) for r in response]
        print(response)

        for res in response:
            question = res['question']
            options = res['options']
            answer = res['answer']

            stb.single_choice(question, options, answer)
