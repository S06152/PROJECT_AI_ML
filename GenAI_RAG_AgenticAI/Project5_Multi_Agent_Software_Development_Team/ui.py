import streamlit as st
import requests

API_URL = "http://localhost:8000/build"

st.title("🤖 AI Software Development Team")

st.write("Describe the software you want the AI team to build.")

user_request = st.text_area(
    "Enter your idea",
    "Build a FastAPI todo application"
)

if st.button("Generate Software"):

    with st.spinner("AI team working..."):

        response = requests.post(
            API_URL,
            json={"user_request": user_request}
        )

        result = response.json()

        st.subheader("📋 Product Specification")
        st.write(result["product_spec"])

        st.subheader("🏗 Architecture")
        st.write(result["architecture"])

        st.subheader("💻 Code")
        st.code(result["code"], language="python")

        st.subheader("🧪 Tests")
        st.code(result["tests"])

        st.subheader("🔍 Code Review")
        st.write(result["review"])