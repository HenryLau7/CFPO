from vllm import LLM, SamplingParams
from .base import LLM_Model
from typing import List, Union, Optional
import os
import logging


class VllmModel(LLM_Model):
    def __init__(
        self,
        model_path: Optional[str] = None,
        max_tokens: int = 256,
        stop: str = '',
        repetition_penalty: float = 1.2,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the VLLM model.

        Args:
            model_path (Optional[str]): Path to the model. Defaults to None.
            max_tokens (int): Maximum number of tokens to generate. Defaults to 256.
            stop (str): Stop sequence for generation. Defaults to ''.
            repetition_penalty (float): Penalty for repetition. Defaults to 1.2.
            logger (Optional[logging.Logger]): Logger object for logging messages.
        """
        self.logger = logger

        # Initialize the VLLM model
        self.llm = LLM(model=model_path)

        self.max_tokens = max_tokens
        self.stop = stop
        self.repetition_penalty = repetition_penalty

    def inference(
        self,
        prompt: Union[str, List[str]],
        use_batch_acceleration: bool = True,
        desc: str = '',
    ) -> Union[str, List[str]]:
        """
        Perform inference using the VLLM model.

        Args:
            prompt (Union[str, List[str]]): Input prompt(s) for the model.
            use_batch_acceleration (bool): Whether to use batch acceleration. Defaults to True.
            desc (str): Description of the inference task for logging.

        Returns:
            Union[str, List[str]]: Generated output(s) from the model.
        """
        # Log the inference call
        if self.logger:
            self.logger.info(f"VLLM | {desc}")

        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0,  # Disable sampling for deterministic output
            repetition_penalty=self.repetition_penalty,
            top_p=0.1,  # Nucleus sampling
            max_tokens=self.max_tokens,
            stop=self.stop,
        )
        
        # Handle batch inference
        if use_batch_acceleration and isinstance(prompt, list):
            batch_size = 512  # Adjust batch size as needed
            gen_output_list = []

            for start_idx in range(0, len(prompt), batch_size):
                end_idx = start_idx + batch_size
                sub_gen_input_list = prompt[start_idx:end_idx]
                sub_gen_output_list = self.llm.generate(sub_gen_input_list, sampling_params, use_tqdm=False)
                gen_output_list.extend(sub_gen_output_list)

            return [item.outputs[0].text for item in gen_output_list]

        # Handle single prompt inference
        elif not use_batch_acceleration and isinstance(prompt, str):
            output = self.llm.generate(prompt, sampling_params, use_tqdm=False)
            return output[0].outputs[0].text

        else:
            raise ValueError(
                "Invalid input: `prompt` must be a list if `use_batch_acceleration` is True, or a string if False."
            )