import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.agents.base_agent import BaseAgent
from src.models.state import DevTeamState

class CodeReviewAgent(BaseAgent):
    """
    CodeReviewAgent is responsible for reviewing source code
    and providing improvement suggestions.

    The agent acts like a Senior Software Engineer and evaluates:
        - Code quality
        - Best practices
        - Possible bugs
        - Performance improvements
        - Readability and maintainability
    """

    def __init__(self, llm):
        """
        Initialize the CodeReviewAgent with a system prompt
        that instructs the LLM to behave as a senior engineer.
        """

        logging.info("Initializing CodeReviewAgent")

        system_prompt = """
        You are a Senior Software Engineer.
        Review the code and give improvements.
        """

        # Initialize the BaseAgent with system prompt
        super().__init__(llm, system_prompt)

        logging.info("CodeReviewAgent initialized successfully")

    def execute(self, state: DevTeamState):
        """
        Executes the code review process.

        Parameters
        ----------
        state : DevTeamState
            Shared state object containing workflow data.
            Expected key:
                - code : Source code to review

        Returns
        -------
        dict
            Dictionary containing review suggestions.
        """

        try:
            logging.info("CodeReviewAgent execution started")

            # Run the LLM review process
            review = self.run(state["code"])

            logging.info("Code review generated successfully")

            # Return result in expected state format
            return {"review": review}

        except Exception as e:
            logging.error("Error occurred during code review process")
            raise CustomException(e, sys)