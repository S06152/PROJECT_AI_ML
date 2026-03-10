import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.agents.base_agent import BaseAgent
from src.models.state import DevTeamState

class ProductManagerAgent(BaseAgent):
    """
    ProductManagerAgent is responsible for converting a user's request
    into a detailed product specification.

    The agent acts like a Product Manager and generates:
        - Feature descriptions
        - API endpoints
        - Functional requirements
        - System requirements
    """

    def __init__(self):
        """
        Initialize the ProductManagerAgent with a system prompt that
        instructs the LLM to behave as a Product Manager.
        """

        logging.info("Initializing ProductManagerAgent")

        system_prompt = """
        You are a Product Manager.
        Write detailed product specifications.
        Include features, endpoints, and requirements.
        """

        # Initialize BaseAgent with the system prompt
        super().__init__(system_prompt)

        logging.info("ProductManagerAgent initialized successfully")

    def execute(self, state: DevTeamState):
        """
        Executes the product specification generation process.

        Parameters
        ----------
        state : DevTeamState
            Shared workflow state containing user input.
            Expected key:
                - user_request : User's original request or idea

        Returns
        -------
        dict
            Dictionary containing the generated product specification.
        """

        try:
            logging.info("ProductManagerAgent execution started")

            # Generate product specification using the LLM
            spec = self.run(state["user_request"])

            logging.info("Product specification generated successfully")

            # Return result in the workflow state format
            return {"product_spec": spec}

        except Exception as e:
            logging.error("Error occurred during product specification generation")
            raise CustomException(e, sys)