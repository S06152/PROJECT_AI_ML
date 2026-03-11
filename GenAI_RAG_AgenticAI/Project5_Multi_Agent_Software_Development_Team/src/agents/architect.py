import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.agents.base_agent import BaseAgent
from src.models.state import DevTeamState

class ArchitectAgent(BaseAgent):
    """
    ArchitectAgent is responsible for generating the high-level
    system architecture based on the provided product specification.

    It uses the BaseAgent's LLM execution capability to produce
    architecture including:
        - Components
        - APIs
        - Technology stack
    """

    def __init__(self, llm):
        """
        Initialize the ArchitectAgent with a system prompt that
        instructs the LLM to behave like a software architect.
        """

        logging.info("Initializing ArchitectAgent")

        system_prompt = """
        You are a Software Architect.
        Design system architecture based on product specs.
        Include components, APIs, and tech stack.
        """

        # Initialize the parent BaseAgent with the system prompt
        super().__init__(llm, system_prompt)

        logging.info("ArchitectAgent initialized successfully")

    def execute(self, state: DevTeamState):
        """
        Executes the architecture design process.

        Parameters
        ----------
        state : DevTeamState
            Shared workflow state containing project information.
            Expected key:
                - product_spec : Product specification document

        Returns
        -------
        dict
            Dictionary containing generated architecture.
        """

        try:
            logging.info("ArchitectAgent execution started")

            # Run LLM to generate architecture
            architecture = self.run(state["product_spec"])

            logging.info("Architecture generation completed successfully")

            return {"architecture": architecture}

        except Exception as e:
            logging.error("Error occurred in ArchitectAgent execution")
            raise CustomException(e, sys)