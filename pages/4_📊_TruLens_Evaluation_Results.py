import streamlit as st

# Streamlit page configuration
st.set_page_config(page_title="📊 Gemini LLM Evaluation with RAG Metrics using TruLens Evals 🌟", layout="wide")

def display_content():
    st.markdown("""
    # 📊 Gemini LLM Evaluation using TruLens Evals based on RAG triad of metrics

    This document presents the evaluation results of the Gemini Language Model (LLM) using TruLens Evals based on the Retrieval Augmented Generation (RAG) triad of metrics. These metrics are crucial for assessing the relevancy of the model's answers to user prompts.

    ## 📊 Overview

    The RAG approach combines the power of retrieval and language generation to provide more accurate and relevant responses. This evaluation focuses on three key metrics:

    1. **Relevance of Retrieved Documents**: Measures how relevant the retrieved documents are to the user prompt.
    2. **Quality of Answer Generation**: Assesses the quality of answers generated based on the retrieved documents.
    3. **Overall Response Coherence**: Evaluates the overall coherence and context-awareness of the final response.

    ## 🔍 Evaluation Metrics

    ### 1️⃣ Relevance of Retrieved Documents
    - **Description**: Analyzes the relevance of documents retrieved by the model in response to a query.
    - **Importance**: Ensures that the foundational information used for generating responses is accurate and pertinent.

    ### 2️⃣ Quality of Answer Generation
    - **Description**: Assesses the accuracy and appropriateness of the answers generated based on the retrieved information.
    - **Importance**: Critical for ensuring the utility and reliability of the responses.

    ### 3️⃣ Overall Response Coherence
    - **Description**: Evaluates how well the generated response integrates the retrieved information coherently.
    - **Importance**: Ensures that the response is not only accurate but also contextually appropriate and coherent.
    """)

    st.markdown("## 📸 Dashboard Screenshots")
    image_paths = [
        "./images/trulens_eval_dashboard.png",
        "./images/truelens_eval_01.png",
        "./images/truelens_eval_02.png",
        "./images/truelens_eval_03.png",
        "./images/truelens_eval_04.png",
        "./images/truelens_eval_05.png",
        "./images/truelens_eval_06.png"
    ]

    for idx, path in enumerate(image_paths, start=1):
        st.markdown(f"### Screenshot {idx}")
        st.image(path, use_column_width=True)
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
    main()
