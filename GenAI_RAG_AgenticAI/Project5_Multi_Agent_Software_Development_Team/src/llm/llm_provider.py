import sys
import os
from src.utils.logger import logging
from src.utils.exception import CustomException
from langchain_groq import ChatGroq
from src.config.config import Config
import streamlit as st

class LLMProvider(Config):
    """
    LLMProvider is responsible for initializing and providing
    the Large Language Model (LLM) instance used across the system.

    This class extends the Config class to access configuration
    values such as:
        - GROQ API key
        - Model name
    """

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
            logging.info("Creating ChatGroq LLM instance")
            st.write("Inside try block of get_llm method")
            st.write("API KEY:",  self.groq_api_key)
            st.write("Model:",  self.model_name)
            # Validate API key before creating LLM
            if not self.groq_api_key:
                logging.error("GROQ_API_KEY is missing. Cannot initialize LLM.")
                raise ValueError("GROQ_API_KEY not found in environment variables")

            # Initialize ChatGroq LLM
            llm = ChatGroq(
                model=self.model_name,
                api_key=os.getenv("GROQ_API_KEY")
            )

            logging.info(f"LLM initialized successfully with model: {self.model_name}")

            return llm

        except Exception as e:
            logging.error("Error occurred while creating ChatGroq LLM instance")
            raise CustomException(e, sys)