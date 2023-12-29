import streamlit as st
import re
import database as db
from OAuthClientLib import *

st.set_page_config(page_title="ESG AI Strategist", page_icon="🌿", layout="wide")

def main():

    col1, col2 = st.columns((2, 1))  

    with col1:
        # Detailed description
        st.markdown("""
        🌿🌍 **ESG AI Strategist** is at the forefront of environmental, social, and governance innovation, leveraging the power of artificial intelligence to redefine sustainability in the corporate world. This pioneering AI-based application offers in-depth insights and actionable strategies for companies and decision-makers, enabling them to not only align with but also excel in ESG practices and Sustainable Development Goals (SDGs). 
        
        🌱🌏 Whether it's navigating the complexities of sustainable practices, crafting robust ESG frameworks, or integrating global SDGs into corporate ethos, ESG AI Strategist is your ultimate ally. With its cutting-edge technology, this tool empowers businesses to make informed, ethical decisions that lead to a sustainable, prosperous future for all.
        """, unsafe_allow_html=True)

        st.markdown("""
        🚀🌟 Embrace the change, drive innovation, and become a leader in the global movement towards a more responsible, eco-friendly, and equitable world.
        """, unsafe_allow_html=True)

        # Key features of the custom multimodal GPT
        key_features = """
        - Leverages LLMs (Gemini pro & Gemini Pro Vision) to redefine sustainability in the corporate sector.
        - Provides in-depth insights for ESG practices and sustainable development goals.
        - Backed by Gemini Pro and Gemini Pro Vision for multimodal capabilities.
        - Analyzes infographics to understand key areas and generate valuable insights.
        - Generates PowerPoint presentations from AI conversations.
        - Features 'TruLens Leaderboard' right in the app to show RAG triad of metrics (answer relevancy, context relevancy and groundedness) in a straemlined way.
        - Offers detailed analysis of user inputs and AI responses.
        - Evaluates performance of Gemini completions using TruLens based on both predefined questions and personalized conversation history.
        """

        with st.expander("**Key Features of ESG AI Strategist GPT**", expanded=True):
            st.markdown(key_features)

        st.subheader("App Demo")

        st.video("https://www.youtube.com/watch?v=M1mBjVC3YMQ")

        

    with col2:
        st.image("./images/esg-gpt.png", caption="Empowering Sustainable Futures")

def is_email_valid(email):
    # Simple regex for validating an email
    pattern = r"^\S+@\S+\.\S+$"
    return re.match(pattern, email)

def is_password_strong(password):
    # Check if the password is at least 6 characters
    if len(password) < 6:
        return False
    return True

def is_email_unique(email):
    # Check if the email is already in the database
    return not db.is_user_exist(email)

def passwords_match(password, repeated_password):
    # Check if both passwords match
    return password == repeated_password

def register_user():
    with st.form("Register User Form"):
        st.subheader("Register User")
        email = st.text_input("Email*")
        name = st.text_input("Name*")
        password = st.text_input("Password*", type="password")
        repeated_password = st.text_input("Repeat Password*", type="password")
        submit_button = st.form_submit_button("Register")

        if submit_button:
            if not is_email_valid(email):
                st.error("Please enter a valid email address.")
                return False
            if not name or name == "":
                st.error("Please enter your name.")
                return False
            if not is_password_strong(password):
                st.error("Password must be at least 6 characters long.")
                return False
            if not passwords_match(password, repeated_password):
                st.error("Passwords do not match.")
                return False
            if not is_email_unique(email):
                st.error("An account with this email already exists.")
                return False
            
            db.insert_user(email, name, password)
            return True
    return False


def login():
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if username and password:
                result, name, email = db.login_user(username, password)
                if result:
                    st.session_state["authentication_status"] = True
                    st.session_state["name"] = name
                    st.session_state["email"] = email
                    st.rerun()
                else:
                    # st.session_state["authentication_status"] = False
                    st.error("Incorrect username or password")
            else:
                st.error("Please enter both username and password")

def logout():
    st.session_state["authentication_status"] = None
    st.session_state["name"] = None
    st.session_state["email"] = None
    st.rerun()

if __name__ == "__main__":
    st.title("ESG AI Strategist (ESG Multimodal GPT)🌍")
    st.caption("🌿🌍 ESG AI Strategist: Revolutionizing sustainable futures with AI-powered insights, guiding companies and decision-makers to embrace and excel in ESG practices and Sustainable Development Goals (SDGs). 💡🚀 Propel your business towards a greener, more responsible tomorrow!")

    if "authentication_status" not in st.session_state or st.session_state["authentication_status"] is None:
        google_login_col, login_text_col, github_login_col = st.columns([4,1,4])
        with google_login_col:
            login_google_oauth()

        with github_login_col:
            login_github_oauth()

    # Initialize session state for authentication
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None

    # Sidebar content
    if st.session_state.get("authentication_status"):
        st.sidebar.subheader(f"""Welcome, {st.session_state.get('name', st.session_state["email"])}""")
        if st.sidebar.button("Logout"):
            logout()  
            st.rerun()

    # Main page content
    if st.session_state["authentication_status"]:
        main() 
    else:
        # Authentication (Login/SignUp) options
        menu = ["Login", "SignUp"]
        choice = st.selectbox("**You can also select to Login or SignUp from the below drop down list**", menu)
        if choice == "Login":
            st.markdown(f"**If you don't have an account, please select the sign-up option from the dropdown list to register with your email address.**")
            login()
        elif choice == "SignUp":
            if register_user():
                st.success("Your account has been registered successfully! You can use your email and password to access the app.", icon="✅")
