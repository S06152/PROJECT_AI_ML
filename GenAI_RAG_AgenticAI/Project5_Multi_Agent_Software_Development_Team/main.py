from fastapi import FastAPI
from pydantic import BaseModel

# import your workflow
from src.graph.workflow_graph import DevTeamWorkflow

app = FastAPI(title="AI Dev Team API")

# initialize workflow
workflow = DevTeamWorkflow()
graph = workflow.build_graph()


class UserRequest(BaseModel):
    user_request: str


@app.get("/")
def home():
    return {"message": "AI Dev Team Running"}


@app.post("/build")
def build_software(request: UserRequest):

    initial_state = {
        "user_request": request.user_request,
        "product_spec": "",
        "architecture": "",
        "code": "",
        "tests": "",
        "review": ""
    }

    result = graph.invoke(initial_state)

    return {
        "product_spec": result.get("product_spec"),
        "architecture": result.get("architecture"),
        "code": result.get("code"),
        "tests": result.get("tests"),
        "review": result.get("review")
    }