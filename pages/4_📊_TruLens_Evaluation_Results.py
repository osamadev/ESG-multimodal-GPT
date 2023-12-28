import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Streamlit page configuration
st.set_page_config(page_title="📊 Gemini LLM Evaluation with RAG Metrics using TruLens Evals 🌟", layout="wide")

def display_content():
    st.markdown("""
# 📊 Gemini LLM Evaluation using TruLens Evals based on RAG triad of metrics
This document presents the evaluation results of the Gemini Language Model (LLM) using TruLens Evals based on the Retrieval Augmented Generation (RAG) triad of metrics. These metrics are crucial for assessing the relevancy of the model's answers to user prompts.

## Overview
RAG combines retrieval and language generation for accurate, relevant responses. The evaluation emphasizes three critical metrics:

### 🔍 Evaluation Metrics
#### 1️⃣ Answer Relevancy (Quality of Answer Generation)
- **Description:** Assesses the precision and appropriateness of generated answers based on retrieved information.
- **Importance:** Key for response utility and reliability, ensuring answers directly address the query's intent.

#### 2️⃣ Context Relevancy (Relevance of Retrieved Documents)
- **Description:** Measures the pertinence of documents retrieved in response to a query.
- **Importance:** Critical for grounding responses in relevant context, ensuring foundational information is pertinent.

#### 3️⃣ Groundedness (Overall Response Coherence)
- **Description:** Evaluates the integration of retrieved information into coherent, context-aware responses.
- **Importance:** Guarantees that the response is contextually appropriate and coherent, blending accuracy with relevancy.

These metrics ensure responses are not only factually accurate but also contextually relevant and grounded in coherent, reliable information.
""")

    st.markdown("## 📸 Dashboard Screenshots")
    image_paths = {
        "./images/Trulens_Leaderboard.png": "TruLens Leaderboard for ESG Multimodal GPT",
        "./images/TruLens_Evaluations_01.png": "TruLens Evaluation Records",
        "./images/Evaluation_Feedback_Functions.png": "Evaluation Feedback Functions",
        "./images/TruLens_Evaluations_02.png": "TruLens Evaluation Relevancy Scores",
        "./images/TruLens_Evaluations_03.png": "Sample Record Details (1)",
        "./images/TruLens_Evaluations_04.png": "Sample Record Details (2)",
        "./images/TruLens_Evaluations_06.png" : "Evaluation Results of RAG Triad of Metrics"
    }

    for image_path, image_title in image_paths.items():
        st.markdown(f"### {image_title}")
        st.image(image_path, use_column_width=True)
        st.markdown("---")

    st.markdown("""
    ## 📝 Key Findings and Insights

   Based on the RAG evaluation of the Gemini LLM, several key findings have emerged:

    - **High Relevance in Document Retrieval**: The Gemini LLM showed a strong ability to retrieve documents highly relevant to the user prompts, indicating robust understanding of query context.
    - **Varied Performance in Answer Generation**: While the LLM was often accurate in generating answers, some responses lacked specificity or depth, suggesting room for improvement in leveraging retrieved information.
    - **Contextual Coherence**: The overall coherence of responses was good, but there were occasional instances where the integration of retrieved information could be more seamless.
    - **Speed and Efficiency**: The response time was generally efficient, though optimization could further enhance the user experience.

    These insights point to a robust foundation in the Gemini LLM's retrieval capabilities, with some areas identified for enhancing answer generation and contextual integration.


    ## 🔚 Conclusions and Next Steps

      The evaluation of the Gemini LLM using RAG metrics has provided valuable insights into its current capabilities and areas for improvement:

    - **Future Enhancements**: Focusing on improving the depth and specificity of generated answers will be a key area of development.
    - **Advanced Contextual Integration**: Enhancing the model's ability to more seamlessly integrate retrieved information into coherent responses.
    - **Continued Monitoring and Evaluation**: Regular evaluations using updated RAG metrics and user feedback will help in continuously improving the model.

    Overall, the Gemini LLM presents a promising foundation with clear pathways for future enhancements to better meet user needs and expectations.

    """)

def main():
    display_content()

if __name__ == "__main__":
    if "authentication_status" not in st.session_state \
    or st.session_state["authentication_status"] == None or st.session_state["authentication_status"] == False:
        switch_page("Home")
    main()
