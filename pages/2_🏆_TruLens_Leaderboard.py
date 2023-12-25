import argparse
import asyncio
import json
import math
import sys

# https://github.com/jerryjliu/llama_index/issues/7244:
asyncio.set_event_loop(asyncio.new_event_loop())

from millify import millify
import numpy as np
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from trulens_eval.db_migration import MIGRATION_UNKNOWN_STR
from trulens_eval.ux.styles import CATEGORY

st.runtime.legacy_caching.clear_cache()

from trulens_eval import Tru, TruChain
from trulens_eval.ux import styles
from trulens_eval.ux.components import draw_metadata

from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import VertexAI
import pinecone
import os
import random
from feedback_functions import load_feedback_functions

st.set_page_config(page_title="Leaderboard", layout="wide")

from trulens_eval.ux.add_logo import add_logo_and_style_overrides

# add_logo_and_style_overrides()

database_url = None


def streamlit_app():
    tru = Tru(database_url=database_url)
    lms = tru.db

    # Set the title and subtitle of the app
    st.title("App Leaderboard")
    st.write(
        "Average feedback values displayed in the range from 0 (worst) to 1 (best)."
    )
    df, feedback_col_names = lms.get_records_and_feedback([])
    feedback_defs = lms.get_feedback_defs()
    feedback_directions = {
        (
            row.feedback_json.get("supplied_name", "") or
            row.feedback_json["implementation"]["name"]
        ): row.feedback_json.get("higher_is_better", True)
        for _, row in feedback_defs.iterrows()
    }

    with st.sidebar:
        if st.button("**Evaluate using Predefined Questions**"):
            filePath = './eval_questions.txt'
            eval_gemini_completions(filePath)
            st.sidebar.success("Feedback functions have been executed successfully!")
            st.rerun()

        if st.button("**Evaluate using Conversation History**  "):
            # Extracting user prompts from the session history
            if 'history' in st.session_state and st.session_state['history']:
                user_prompts = [user_text for user_text, _ in st.session_state['history']]
                # Run the evaluation feedback functions
                if eval_gemini_completions(user_prompts):
                    st.sidebar.success("Feedback functions have been executed successfully!")
                else:
                    st.sidebar.warning("The conversation history is empty!")
            else:
                st.sidebar.warning("The conversation history is empty.")

    if df.empty:
        st.write("No records yet...")
        return

    df = df.sort_values(by="app_id")

    if df.empty:
        st.write("No records yet...")

    apps = list(df.app_id.unique())
    st.markdown("""---""")

    for app in apps:
        app_df = df.loc[df.app_id == app]
        if app_df.empty:
            continue
        app_str = app_df["app_json"].iloc[0]
        app_json = json.loads(app_str)
        metadata = app_json.get("metadata")
        # st.text('Metadata' + str(metadata))
        st.header(app, help=draw_metadata(metadata))
        app_feedback_col_names = [
            col_name for col_name in feedback_col_names
            if not app_df[col_name].isna().all()
        ]
        col1, col2, col3, col4, *feedback_cols, col99 = st.columns(
            5 + len(app_feedback_col_names)
        )
        latency_mean = (
            app_df["latency"].
            apply(lambda td: td if td != MIGRATION_UNKNOWN_STR else None).mean()
        )

        # app_df_feedback = df.loc[df.app_id == app]

        col1.metric("Records", len(app_df))
        col2.metric(
            "Average Latency (Seconds)",
            (
                f"{millify(round(latency_mean, 5), precision=2)}"
                if not math.isnan(latency_mean) else "nan"
            ),
        )
        col3.metric(
            "Total Cost (USD)",
            f"${millify(round(sum(cost for cost in app_df.total_cost if cost is not None), 5), precision = 2)}",
        )
        col4.metric(
            "Total Tokens",
            millify(
                sum(
                    tokens for tokens in app_df.total_tokens
                    if tokens is not None
                ),
                precision=2
            ),
        )

        for i, col_name in enumerate(app_feedback_col_names):
            mean = app_df[col_name].mean()

            st.write(
                styles.stmetricdelta_hidearrow,
                unsafe_allow_html=True,
            )

            higher_is_better = feedback_directions.get(col_name, True)

            if "distance" in col_name:
                feedback_cols[i].metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta_color="normal"
                )
            else:
                cat = CATEGORY.of_score(mean, higher_is_better=higher_is_better)
                feedback_cols[i].metric(
                    label=col_name,
                    value=f"{round(mean, 2)}",
                    delta=f"{cat.icon} {cat.adjective}",
                    delta_color=(
                        "normal" if cat.compare(
                            mean, CATEGORY.PASS[cat.direction].threshold
                        ) else "inverse"
                    ),
                )

        with col99:
            if st.button("Select App", key=f"app-selector-{app}"):
                st.session_state.app = app
                switch_page("TruLens Evaluations")

        st.markdown("""---""")


                        
def eval_gemini_completions(filePath: str):
    tru = Tru()
    tru.reset_database()
    # Initialize TruChain with qa_chain and feedback functions
    (f_groundedness, f_qa_relevance, f_context_relevance, f_hate, f_violent, f_selfharm, f_maliciousness) = \
    load_feedback_functions()
    tru_recorder = TruChain(create_coversational_chain(),
                            app_id='ESG-GPT',
                            feedbacks=[f_qa_relevance, f_context_relevance, f_groundedness, f_violent])
    
    # Read questions from a file
    with open(filePath, 'r') as file:
        questions = file.readlines()

    for question in questions:
        question = question.strip()
        tru_recorder(question)

def eval_gemini_completions(user_questions):
    tru = Tru()
    tru.reset_database()
    # Initialize TruChain with qa_chain and feedback functions
    (f_groundedness, f_qa_relevance, f_context_relevance, f_hate, f_violent, f_selfharm, f_maliciousness) = \
    load_feedback_functions()
    tru_recorder = TruChain(create_coversational_chain(),
                            app_id='ESG-GPT',
                            feedbacks=[f_qa_relevance, f_context_relevance, f_groundedness, f_violent])
    
    # Check if user questions list is empty
    if not user_questions:
        return False
    else:  
        # Randomly select up to three questions
        selected_questions = random.sample(user_questions, min(3, len(user_questions)))
        for question in selected_questions:
            tru_recorder(question.strip())
        return True


def create_coversational_chain():
    index = initialize_pinecone()
    model_name = 'text-embedding-ada-002'
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    embed = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
    vectorstore = Pinecone(index, embed, "text")

    llm = VertexAI(model_name="gemini-pro")

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history', input_key='input', output_key='output', k=5, return_messages=True)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qa

def initialize_pinecone():
    index_name = 'esg-index'
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric='cosine', dimension=1536)

    index = pinecone.Index(index_name)
    return index

# Define the main function to run the app
def main():
    streamlit_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url", default=None)

    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently, streamlit prevents the program from exiting normally,
        # so we have to do a hard exit.
        sys.exit(e.code)

    database_url = args.database_url

    main()