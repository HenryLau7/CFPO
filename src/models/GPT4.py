from .base import LLM_Model
from openai import AzureOpenAI
import os
import time
from typing import Optional
import logging


class GPT4Model(LLM_Model):
    def __init__(self, max_tokens: int, logger: Optional[logging.Logger] = None):
        """
        Initialize the GPT-4 model with Azure OpenAI credentials.

        Args:
            max_tokens (int): Maximum number of tokens to generate.
            logger (Optional[logging.Logger]): Logger object for logging messages.
        """
        self.max_tokens = max_tokens
        self.logger = logger

        # Validate environment variables
        endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', '')
        api_key = os.environ.get('AZURE_OPENAI_API_KEY', '')
        if not endpoint or not api_key:
            raise ValueError(
                "Please set the environment variables AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
            )

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-05-01-preview",
        )

    def inference(self, prompt: str, temperature: float, desc: str = '') -> str:
        """
        Perform inference using the GPT-4 model.

        Args:
            prompt (str): The input prompt for the model.
            temperature (float): Sampling temperature for the model.
            desc (str): Description of the inference task for logging.

        Returns:
            str: The generated response from the model.
        """
        # Log the inference call
        if self.logger:
            self.logger.info(f"GPT4 | {desc} | Temperature: {temperature}")

        messages = [{'role': 'user', 'content': prompt}]
        response = ""
        timeout = 5

        while not response:
            try:
                time.sleep(timeout)
                completion = self.client.chat.completions.create(
                    model="gpt-4",  # Deployment name
                    messages=messages,
                    seed=42,  # Ensure reproducibility
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )
                response = completion.choices[0].message.content
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error during GPT-4 inference: {e}")

                # Exponential backoff for retries
                timeout = min(timeout * 2, 1024)  # Cap timeout at 1024 seconds
                if timeout > 120:
                    timeout = 1  # Reset timeout if it grows too large

                if self.logger:
                    self.logger.info(f"Retrying after {timeout} seconds...")

                if timeout > 1024:
                    if self.logger:
                        self.logger.error("Max retries exceeded. Aborting inference.")
                    break

        return response