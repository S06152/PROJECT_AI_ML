import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_groq import ChatGroq
from src.config.config import Config

class LLMProvider(Config):
    """
    LLMProvider is responsible for initializing and providing
    the Large Language Model (LLM) instance used across the system.

    This class extends the Config class to access configuration
    values such as:
        - GROQ API key
        - Model name
    """

    # Singleton instance to avoid multiple LLM creations
    _llm_instance = None 

    def __init__(self):
        """
        Initialize the LLMProvider by loading configuration settings.
        """

        try:
            logging.info("Initializing LLMProvider")

            # Load configuration from Config class
            super().__init__()
            
            logging.info("LLMProvider initialized successfully")

        except Exception as e:
            logging.error("Error occurred while initializing LLMProvider")
            raise CustomException(e, sys)

    def get_llm(self):
        """
        Create and return the ChatGroq LLM instance.

        Returns
        -------
        ChatGroq
            Configured Groq LLM client used by agents.
        """

        try:
            logging.info("Request received to get LLM instance.")

            # Return cached instance if already created
            if LLMProvider._llm_instance is not None:
                logging.info("Returning existing LLM instance (Singleton pattern).")
                return LLMProvider._llm_instance
            
            logging.info("Creating new ChatGroq LLM instance.")

            # Validate API key
            if not self.groq_api_key:
                logging.error("GROQ_API_KEY is missing. Cannot initialize LLM.")
                raise ValueError("GROQ_API_KEY not found in environment variables")

            # Validate Model Name
            if not self.model_name:
                logging.error("Model name is missing in configuration.")
                raise ValueError("Model name not provided.")
            
            logging.info(f"Initializing Groq LLM with model: {self.model_name}")

            # Initialize LLM
            llm = ChatGroq(
                model=self.model_name,
                api_key=self.groq_api_key
            )

            # Store singleton instance
            LLMProvider._llm_instance = llm

            logging.info("ChatGroq LLM instance created and cached successfully.")

            return llm

        except Exception as e:
            logging.error("Error occurred while creating ChatGroq LLM instance")
            raise CustomException(e, sys)