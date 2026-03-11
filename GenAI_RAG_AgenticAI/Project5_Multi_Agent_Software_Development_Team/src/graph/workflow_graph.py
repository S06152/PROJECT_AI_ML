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
from src.llm.llm_provider import LLMProvider

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
        Initialize all agents and shared resources required for the workflow.
        """

        try:
            logging.info("Initializing DevTeamWorkflow...")

            # Initialize LLM provider
            logging.info("Initializing LLM Provider...")
            llm_provider = LLMProvider()

            logging.info("Fetching LLM instance from provider...")
            llm = llm_provider.get_llm()

            logging.info("LLM instance obtained successfully.")

            # Initialize agents with shared LLM instance
            logging.info("Initializing ProductManagerAgent...")
            self.pm = ProductManagerAgent(llm)

            logging.info("Initializing ArchitectAgent...")
            self.arch = ArchitectAgent(llm)

            logging.info("Initializing DeveloperAgent...")
            self.dev = DeveloperAgent(llm)

            logging.info("Initializing QAAgent...")
            self.qa = QAAgent(llm)

            logging.info("Initializing CodeReviewAgent...")
            self.review = CodeReviewAgent(llm)

            logging.info("All agents initialized successfully.")

        except Exception as e:
            logging.error("Failed to initialize DevTeamWorkflow agents.")
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
            logging.info("Building DevTeam workflow graph...")

            # Initialize StateGraph
            logging.info("Creating StateGraph with DevTeamState...")
            graph = StateGraph(DevTeamState)

            # Register agent nodes
            logging.info("Adding ProductManagerAgent node...")
            graph.add_node("ProductManagerAgent", self.pm.execute)

            logging.info("Adding ArchitectAgent node...")
            graph.add_node("ArchitectAgent", self.arch.execute)

            logging.info("Adding DeveloperAgent node...")
            graph.add_node("DeveloperAgent", self.dev.execute)

            logging.info("Adding QAAgent node...")
            graph.add_node("QAAgent", self.qa.execute)

            logging.info("Adding CodeReviewAgent node...")
            graph.add_node("CodeReviewAgent", self.review.execute)

            logging.info("All agent nodes added successfully.")

            # Define workflow edges
            logging.info("Defining workflow execution order...")

            graph.add_edge(START, "ProductManagerAgent")
            graph.add_edge("ProductManagerAgent", "ArchitectAgent")
            graph.add_edge("ArchitectAgent", "DeveloperAgent")
            graph.add_edge("DeveloperAgent", "QAAgent")
            graph.add_edge("QAAgent", "CodeReviewAgent")
            graph.add_edge("CodeReviewAgent", END)

            logging.info("Workflow edges defined successfully.")

            # Compile graph
            logging.info("Compiling DevTeam workflow graph...")
            compiled_graph = graph.compile()

            logging.info("DevTeam workflow compiled successfully.")

            return compiled_graph

        except Exception as e:
            logging.error("Error occurred while building DevTeam workflow graph.")
            raise CustomException(e, sys)