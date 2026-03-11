import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.llm.llm_provider import LLMProvider
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class BaseAgent:
    """
    BaseAgent provides the common functionality required for all AI agents.

    Responsibilities:
    - Initialize the LLM provider
    - Build the prompt template
    - Execute the LangChain pipeline
    - Return the parsed LLM response

    Other agents such as ArchitectAgent, DeveloperAgent, etc.
    should inherit from this class.
    """

    def __init__(self, llm, system_prompt: str):
        """
        Initialize the BaseAgent with a system prompt.

        Parameters
        ----------
        system_prompt : str
            The system instruction that defines the behavior of the agent.
        """

        try:
            logging.info("Initializing BaseAgent")

            # Initialize LLM 
            self.llm = llm

            # Store system prompt for later execution
            self.system_prompt = system_prompt

            logging.info("BaseAgent initialized successfully")

        except Exception as e:
            logging.error("Error while initializing BaseAgent")
            raise CustomException(e, sys)

    def run(self, input_text: str) -> str:
        """
        Executes the agent by sending the input to the LLM.

        Parameters
        ----------
        input_text : str
            Input text (e.g., product specification, requirements, etc.)

        Returns
        -------
        str
            Generated response from the LLM.
        """

        try:
            logging.info("Agent execution started")

            # Create prompt template with system and user messages
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    ("human", "{input_text}")
                ]
            )

            logging.info("Prompt template created successfully")

            # Build LangChain pipeline
            chain = prompt | self.llm | StrOutputParser()

            logging.info("LLM chain constructed")

            # Execute the chain
            response = chain.invoke({"input_text": input_text})

            logging.info("LLM response generated successfully")

            return response

        except Exception as e:
            logging.error("Error occurred during agent execution")
            raise CustomException(e, sys)