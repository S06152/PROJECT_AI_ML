import streamlit as st

# import your workflow
from src.graph.workflow_graph import DevTeamWorkflow


st.title("🤖 AI Software Development Team")

st.write("Describe the software you want to build.")

user_request = st.text_area(
    "Enter software idea",
    "Build a FastAPI todo application"
)


if st.button("Generate Software"):

    with st.spinner("AI team working..."):

        workflow = DevTeamWorkflow()
        graph = workflow.build_graph()

        initial_state = {
            "user_request": user_request,
            "product_spec": "",
            "architecture": "",
            "code": "",
            "tests": "",
            "review": ""
        }

        result = graph.invoke(initial_state)

        st.subheader("📋 Product Specification")
        st.write(result["product_spec"])

        st.subheader("🏗 Architecture")
        st.write(result["architecture"])

        st.subheader("💻 Code")
        st.code(result["code"], language="python")

        st.subheader("🧪 Tests")
        st.code(result["tests"])

        st.subheader("🔎 Code Review")
        st.write(result["review"])