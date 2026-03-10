import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langgraph.graph import StateGraph, START, END
from src.models.state import DevTeamState
from src.agents.product_manager import ProductManagerAgent
from src.agents.architect import ArchitectAgent
from src.agents.developer import DeveloperAgent
from src.agents.qa import QAAgent
from src.agents.code_reviewer import CodeReviewAgent

class DevTeamWorkflow:
    """
    DevTeamWorkflow orchestrates the entire AI development team pipeline.

    This workflow connects multiple AI agents using LangGraph to simulate
    a real software development lifecycle.

    Pipeline Flow:
        User Request
            ↓
        ProductManagerAgent
            ↓
        ArchitectAgent
            ↓
        DeveloperAgent
            ↓
        QAAgent
            ↓
        CodeReviewAgent
            ↓
        END
    """

    def __init__(self):
        """
        Initialize all agents required for the workflow.
        """

        try:
            logging.info("Initializing DevTeamWorkflow")

            # Initialize all agents
            self.pm = ProductManagerAgent()
            self.arch = ArchitectAgent()
            self.dev = DeveloperAgent()
            self.qa = QAAgent()
            self.review = CodeReviewAgent()

            logging.info("All agents initialized successfully")

        except Exception as e:
            logging.error("Error occurred while initializing DevTeamWorkflow agents")
            raise CustomException(e, sys)

    def build_graph(self):
        """
        Build and compile the LangGraph workflow for the development team.

        Returns
        -------
        CompiledGraph
            A compiled LangGraph workflow ready for execution.
        """

        try:
            logging.info("Building DevTeam workflow graph")

            # Create a StateGraph using DevTeamState
            graph = StateGraph(DevTeamState)

            logging.info("Adding agent nodes to workflow")

            # Register all agents as nodes in the workflow
            graph.add_node("ProductManagerAgent", self.pm.execute)
            graph.add_node("ArchitectAgent", self.arch.execute)
            graph.add_node("DeveloperAgent", self.dev.execute)
            graph.add_node("QAAgent", self.qa.execute)
            graph.add_node("CodeReviewAgent", self.review.execute)

            logging.info("Defining workflow execution order")

            # Define execution flow
            graph.add_edge(START, "ProductManagerAgent")
            graph.add_edge("ProductManagerAgent", "ArchitectAgent")
            graph.add_edge("ArchitectAgent", "DeveloperAgent")
            graph.add_edge("DeveloperAgent", "QAAgent")
            graph.add_edge("QAAgent", "CodeReviewAgent")
            graph.add_edge("CodeReviewAgent", END)

            logging.info("Compiling DevTeam workflow graph")

            # Compile the graph for execution
            compiled_graph = graph.compile()

            logging.info("DevTeam workflow compiled successfully")

            return compiled_graph

        except Exception as e:
            logging.error("Error occurred while building DevTeam workflow graph")
            raise CustomException(e, sys)