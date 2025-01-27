import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.prompt import Prompt

task_instruction = "You are a commonsense helper. I will provide several examples and a presented question. Your goal is to pick the most reasonable answer among the given options for the current question. Please respond with the corresponding label (A/B/C/D) for the correct answer."

task_detail = ""

output_format = ""

example_hinter = "Here are some examples:\n"
examples = [
    {
        "question":"Forests have been cut and burned so that the land can be used to raise crops. Which consequence does this activity have on the atmosphere of Earth?",
        "choices": [ "It reduces the amount of carbon dioxide production", "It reduces the production of oxygen", "It decreases the greenhouse effect", "It decreases pollutants in the air" ],
        "answer": "B"
    }
]

cot_hinter = ""
query_part = """
{{ query }}
"""

from src.mutators.format_search_pool.prompt_renderer import direct_joint_renderer, direct_joint_extractor
from mutators.format_search_pool.query_format import multiple_choice_query_format_pool
from src.mutators.format_search_pool.query_format.multiple_choice import QA_renderer_2, QA_extractor_2 

prompt = Prompt(
    round = 0,
    task = 'MultipleChoice',
    query_part=query_part,
    task_instruction = task_instruction,
    task_detail = task_detail,
    output_format = output_format,
    example_hinter = example_hinter,
    examples = examples,
    prompt_renderer_fn = direct_joint_renderer,
    prompt_extract_fn = direct_joint_extractor,
    query_renderer_fn= QA_renderer_2,
    query_extract_fn = QA_extractor_2,
    cot_hinter = cot_hinter,
)