import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import os
from dotenv import load_dotenv
import streamlit as st

class Config:
    """
    Config class is responsible for loading environment variables
    and providing configuration values required across the application.

    Responsibilities:
        - Load environment variables from .env file
        - Provide access to API keys
        - Store model configuration
    """

    def __init__(self, model_name: str = "openai/gpt-oss-safeguard-20b"):
        """
        Initialize configuration settings.

        Parameters
        ----------
        model_name : str
            Default LLM model used by the system.
        """

        try:
            logging.info("Initializing application configuration")

            # Load environment variables from .env file
            load_dotenv()

            logging.info(".env file loaded successfully")

            # Retrieve Groq API key from environment variables
            self.groq_api_key = os.getenv("GROQ_API_KEY")
            st.write("API KEY:",  self.groq_api_key)
            if not self.groq_api_key:
                logging.warning("GROQ_API_KEY not found in environment variables")

            # Store model configuration
            self.model_name = model_name

            logging.info(f"Model configured: {self.model_name}")

        except Exception as e:
            logging.error("Error occurred while initializing configuration")
            raise CustomException(e, sys)