import streamlit as st

# Set the page config for a custom title and favicon (optional)
st.set_page_config(page_title="ESG AI Strategist", page_icon="🌿", layout="wide")

def main():
    # App main title
    st.title("ESG AI Strategist 🌍")

    # Use columns to layout the content elegantly
    col1, col2 = st.columns((2, 1))  # Adjust the ratio as needed

    with col1:
        # Detailed description
        st.markdown("""
        🌿🌍 **ESG AI Strategist** is at the forefront of environmental, social, and governance innovation, leveraging the power of artificial intelligence to redefine sustainability in the corporate world. This pioneering AI-based application offers in-depth insights and actionable strategies for companies and decision-makers, enabling them to not only align with but also excel in ESG practices and Sustainable Development Goals (SDGs). 
        
        🌱🌏 Whether it's navigating the complexities of sustainable practices, crafting robust ESG frameworks, or integrating global SDGs into corporate ethos, ESG AI Strategist is your ultimate ally. With its cutting-edge technology, this tool empowers businesses to make informed, ethical decisions that lead to a sustainable, prosperous future for all.
        """, unsafe_allow_html=True)

    with col2:
        # Optional: Add an image or additional content
        st.image("./images/esg-gpt.png", caption="Empowering Sustainable Futures")

    # Additional content or footer
    st.markdown("""
    🚀🌟 Embrace the change, drive innovation, and become a leader in the global movement towards a more responsible, eco-friendly, and equitable world.
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
