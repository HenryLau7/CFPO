from .base import BaseMutator
from utils import parse_tagged_text, stringify_dict
import math
import random
import inspect
from typing import List, Tuple, Optional, Dict
import re

class FormatMutator(BaseMutator):
    def __init__(
        self,
        mutation_llm,
        task,
        COMPONENT_KEYS: List[str],  # ['PROMPT_FORMAT', 'QUERY_FORMAT']
        prompt_history,
        search_pool: Dict,  # A dict: search_pool['prompt'], search_pool['query']
        logger=None,
    ):
        super().__init__(mutation_llm, task, COMPONENT_KEYS)
        self.task = task
        self.knowledge_components = COMPONENT_KEYS
        self.prompt_history = prompt_history
        self.format_pool = {'PROMPT_FORMAT': None, 'QUERY_FORMAT': None}
        self.search_pool = search_pool
        self.logger = logger
        self._init_knowledge_pool_for_format()

    def _init_knowledge_pool_for_format(self):
        """Initialize the knowledge pool for formats."""
        self.format_pool["PROMPT_FORMAT"] = {
            fn[0]: {'confidence_score': 0, 'chosen_count': 0, 'uct_score': 0, 'eval_score': 0}
            for fn in self.search_pool["prompt"]
        }
        self.format_pool["QUERY_FORMAT"] = {
            fn[0]: {'confidence_score': 0, 'chosen_count': 0, 'uct_score': 0, 'eval_score': 0}
            for fn in self.search_pool["query"]
        }

    def __call__(self, prompts: List, num_formats_apply: int, round: int, method: str = 'UCT') -> List:
        """Generate new prompts by mutating formats."""
        if round == 0:
            return []

        if round == 2:
            new_prompts = []
            for prompt in prompts:
                new_prompts_per_prompt = [prompt]
                new_prompts_per_prompt += self.traverse_format(prompt, round)
                new_prompts.append(new_prompts_per_prompt)
            return new_prompts

        prompt_formats, query_formats = self._generate_and_apply_formats(round, num_formats_apply, method)
        new_prompts = self._apply_formats_to_prompts(prompts, prompt_formats, query_formats, round)

        return new_prompts

    def traverse_format(self, prompt, round: int) -> List:
        """Traverse all formats and generate new prompts."""
        self.logger.info(f"\n================ In Round {round} Traverse all formats ================")
        current_prompt_format = prompt.prompt_format
        current_query_format = prompt.query_format
        new_prompts = []

        for prompt_format in self.search_pool['prompt']:
            for query_format in self.search_pool['query']:
                if prompt_format != current_prompt_format or query_format != current_query_format:
                    component_keys, component_contents = [], []
                    if prompt_format != current_prompt_format:
                        component_keys.append("PROMPT_FORMAT")
                        component_contents.append(prompt_format)
                    if query_format != current_query_format:
                        component_keys.append("QUERY_FORMAT")
                        component_contents.append(query_format)

                    new_prompt = prompt.generate(
                        round=round,
                        component_key=component_keys,
                        component_content=component_contents,
                        action_desc="traverse",
                        reason=None,
                    )
                    new_prompts.append(new_prompt)

        return new_prompts

    def _generate_and_apply_formats(self, round: int, num_formats_apply: int, method: str) -> Tuple[List, List]:
        """Generate and apply new formats."""
        prompt_formats, query_formats = [], []

        # Generate new formats
        new_formats = self.generate_new_format()
        if new_formats:
            bool_prompt, _ = verify.verify_format('PROMPT_FORMAT', new_formats[0])
            bool_query, _ = verify.verify_format('QUERY_FORMAT', new_formats[1])

            if bool_prompt:
                prompt_formats.append(new_formats[0])
            else:
                self.logger.warning(f"Generated prompt format is invalid, format name: {new_formats[0][0].__name__}")

            if bool_query:
                query_formats.append(new_formats[1])
            else:
                self.logger.warning(f"Generated query format is invalid, format name: {new_formats[1][0].__name__}")

        # Apply knowledge-based formats
        new_formats = self.apply_knowledge(num_formats_apply, round, method)
        prompt_formats.extend(new_formats[0])
        query_formats.extend(new_formats[1])

        return prompt_formats, query_formats

    def _apply_formats_to_prompts(self, prompts: List, prompt_formats: List, query_formats: List, round: int) -> List:
        """Apply formats to prompts and generate new prompts."""
        new_prompts = []
        for prompt in prompts:
            new_prompts_per_prompt = [prompt]

            for prompt_format in prompt_formats:
                new_prompt = prompt.generate(
                    round=round,
                    component_key=["PROMPT_FORMAT"],
                    component_content=[prompt_format],
                    action_desc="format",
                    reason=None,
                )
                try:
                    if str(new_prompt) and str(new_prompt) != str(prompt):
                        new_prompts_per_prompt.append(new_prompt)
                except Exception as e:
                    self.logger.error(f"Error generating prompt with prompt format {prompt_format[0].__name__}: {e}")
                    continue

            for query_format in query_formats:
                new_prompt = prompt.generate(
                    round=round,
                    component_key=["QUERY_FORMAT"],
                    component_content=[query_format],
                    action_desc="format",
                    reason=None,
                )
                try:
                    if str(new_prompt) and str(new_prompt) != str(prompt):
                        new_prompts_per_prompt.append(new_prompt)
                except Exception as e:
                    self.logger.error(f"Error generating prompt with query format {query_format[0].__name__}: {e}")
                    continue

            new_prompts.append(new_prompts_per_prompt)

        return new_prompts

    def generate_new_format(self, exist: bool = True) -> Optional[Tuple]:
        """Generate new formats for PROMPT_FORMAT and QUERY_FORMAT."""
        if exist and 'generated_prompt' in self.search_pool:
            generated_prompt_format = self.search_pool['generated_prompt'].pop(0)
            generated_query_format = self.search_pool['generated_query'].pop(0)
            self.search_pool['prompt'].append(generated_prompt_format)
            self.search_pool['query'].append(generated_query_format)
        else:
            generated_prompt_format = self._generate_format_prompt("PROMPT_FORMAT")
            generated_query_format = self._generate_format_prompt("QUERY_FORMAT")

        if generated_prompt_format and generated_query_format:
            self.logger.info(f"Generated formats:\nPrompt format: {generated_prompt_format}\nQuery format: {generated_query_format}")
            self.format_pool["PROMPT_FORMAT"][generated_prompt_format[0]] = {'confidence_score': 0, 'chosen_count': 0, 'uct_score': 0, 'eval_score': 0}
            self.format_pool["QUERY_FORMAT"][generated_query_format[0]] = {'confidence_score': 0, 'chosen_count': 0, 'uct_score': 0, 'eval_score': 0}
            return (generated_prompt_format, generated_query_format)
        return None

    def apply_knowledge(self, num_prompt: int, round: int, method: str = 'UCT') -> Tuple[List, List]:
        """Apply knowledge-based formats."""
        if method == "UCT":
            new_prompt_formats = sorted(
                self.format_pool["PROMPT_FORMAT"].keys(),
                key=lambda k: self.format_pool["PROMPT_FORMAT"][k]['uct_score'],
                reverse=True
            )[:num_prompt]
            new_query_formats = sorted(
                self.format_pool["QUERY_FORMAT"].keys(),
                key=lambda k: self.format_pool["QUERY_FORMAT"][k]['uct_score'],
                reverse=True
            )[:num_prompt]
        elif method == "Random":
            new_prompt_formats = random.sample(list(self.format_pool["PROMPT_FORMAT"].keys()), num_prompt)
            new_query_formats = random.sample(list(self.format_pool["QUERY_FORMAT"].keys()), num_prompt)

        self.logger.info(f"Selected formats:\nPrompt formats: {new_prompt_formats}\nQuery formats: {new_query_formats}")
        return (new_prompt_formats, new_query_formats)

    def _generate_format_prompt(self, component_key: str) -> Optional[Tuple]:
        """Generate a new format for the given component key."""
        if component_key == "PROMPT_FORMAT":
            return self._generate_prompt_format()
        elif component_key == "QUERY_FORMAT":
            return self._generate_query_format()
        return None

    def _generate_prompt_format(self) -> Optional[Tuple]:
        """Generate a new PROMPT_FORMAT."""
        format_fn_desc = []
        for key, content in self.search_pool['prompt_desc'].items():
            format_fn_desc.append((key.__name__[:-9], content))
        format_fn_desc_string = "\n".join([f"{name}: {desc}" for name, desc in format_fn_desc])

        # Task-specific meta prompt
        if self.task.__class__.__name__ in ["MATHTask", "GSM8KTask"]:
            task_specific_instruction = "Ensure the format is suitable for mathematical problem-solving and includes clear instructions for step-by-step reasoning."
        else:
            task_specific_instruction = "Ensure the format is clear, structured, and aligned with commonly used prompt formats."

        prompt_to_generate_format = f"""{self._get_meta_prompt_header()}

        The whole prompt is  \"\"\"{str(self.prompt_history.current_prompt).strip()}\"\"\"

        We have some preset PROMPT_FORMAT candidates, here are our whole search pool:
        {format_fn_desc_string}
        
        Here are two examples from our PROMPT_FORMAT candidates as for your reference:
        <Format name: markdown>
        ##### Task Instruction
        {{TASK_INSTRUCTION}}
        
        ##### Task Detail
        {{TASK_DETAIL}}

        ##### Output Format
        {{OUTPUT_FORMAT}}

        ##### Examples
        {{EXAMPLES}}

        <Format name: xml>
        <TaskInstruction>{{TASK_INSTRUCTION}}</TaskInstruction>
        <TaskDetail>{{TASK_DETAIL}}</TaskDetail>
        <OutputFormat>{{OUTPUT_FORMAT}}</OutputFormat>
        <Examples>{{EXAMPLES}}</Examples>

        Please generate ONE new format for the PROMPT_FORMAT segment, its description and render the {{TASK_INSTRUCTION}}, {{TASK_DETAIL}}, {{OUTPUT_FORMAT}} and {{EXAMPLES}} segments using this new format. The new format could either be distinct from the existing formats, or a variation of an existing format. 
        
        If you choose a completely new format, ensure that the new format is conventional, structured, and aligned with commonly used prompt formats. Avoid overly creative or unconventional formats that deviate significantly from standard practices. 

        If it's a variation of an existing format, the variation can change the order of the segments, or drop some segments.

        {task_specific_instruction}

        The format name should only include alphanumeric characters and underscores. Special characters such as `|`, `!`, `#`, `@`, and spaces should be avoided. 
        
        Please encapsulate the new prompt format using the following format:
        
        <START>
        <Format name: [format name]>
        <Description: [format description]>
        [The rendered segments rendered by the newly generated format]
        <END>
        """

        prompt_to_generate_format = '\n'.join([line.lstrip() for line in prompt_to_generate_format.split('\n')])
        response = self.mutation_llm.inference(prompt_to_generate_format, desc="generate prompt format", temperature=self.temperature)

        new_formats = self._parse_format(parse_tagged_text(response, "<START>", "<END>"))

        if len(new_formats) == 0:
            return None
        return new_formats[0]

    def _generate_query_format(self) -> Optional[Tuple]:
        """Generate a new QUERY_FORMAT."""
        example = {
            "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is: 6."
        }
        rendered_example_1 = self.search_pool['query'][0][0](example["question"], example["answer"], self.prompt_history.current_prompt.cot_hinter)
        rendered_example_2 = self.search_pool['query'][3][0](example["question"], example["answer"], self.prompt_history.current_prompt.cot_hinter)

        format_fn_desc = []
        for key, content in self.search_pool['query_desc'].items():
            format_fn_desc.append((key.__name__[15:], content))
        format_fn_desc_string = "\n".join([f"{name}: {desc}" for name, desc in format_fn_desc])

        # Task-specific meta prompt
        if self.task.__class__.__name__ in ["MATHTask", "GSM8KTask"]:
            task_specific_instruction = "Ensure the format is suitable for mathematical problem-solving and includes clear instructions for step-by-step reasoning."
        else:
            task_specific_instruction = "Ensure the format is clear, structured, and aligned with commonly used query formats."

        prompt_to_generate_format = f"""{self._get_meta_prompt_header()}

        The whole prompt is  \"\"\"{str(self.prompt_history.current_prompt).strip()}\"\"\"

        We have some preset QUERY_FORMAT candidates, here are our whole search pool:
        {format_fn_desc_string}
        
        Here are two examples from our QUERY_FORMAT candidates as for your reference:
        <Format name: QA_0_space>
        {rendered_example_1}

        <Format name: IR>
        {rendered_example_2}

        Please generate ONE new format for the QUERY_FORMAT segment, its description and render the provided example using this new format. The new format could either be a completely new format or a variation of an existing format. 
        
        If you choose to generate a completely new format, please ensure that the new format is conventional, structured, and aligned with commonly used query formats. Avoid overly creative or unconventional formats that deviate significantly from standard practices. The new format should be distinct from the existing formats. 

        The variation can focus on two parts, CASING and SEPARATOR:

        CASING refers to both the capitalization of the text (e.g., f(x) = x.title(), f(x) = x.upper(), f(x) = x.lower()) and the specific wording or phrasing used (e.g., changing "question" to "instruction" or "input"). 

        SEPARATOR: the punctuation or symbols used to separate the question and answer, there are some candidates as for your reference {{'', ' ', '\\n', '--', ';\\n', ' ||', '<sep>', ' \\n', ':', '.'}}.
        
        {task_specific_instruction}

        The format name should only include alphanumeric characters and underscores. Special characters such as `|`, `!`, `#`, `@`, and spaces should be avoided.
        
        Please encapsulate the new query format using the following format:
        
        <START>
        <Format name: [format name]>
        <Description: [format description]>
        [The example rendered by the newly generated format]
        <END>
        """

        prompt_to_generate_format = '\n'.join([line.lstrip() for line in prompt_to_generate_format.split('\n')])
        response = self.mutation_llm.inference(prompt_to_generate_format, desc="generate query format", temperature=self.temperature)

        new_formats = self._parse_format(parse_tagged_text(response, "<START>", "<END>"))

        if len(new_formats) == 0:
            return None
        return new_formats[0]

    def _get_meta_prompt_header(self) -> str:
        """Get the meta prompt header for format generation."""
        return f"""I'm trying to write a prompt to {self.task.task_intention}.
        The ultimate aim is to create a prompt that is clear, structured, and efficient, leading to accurate responses from the AI model. The structure of the prompt includes several essential elements: {self.component_desc}""".strip()

    def _update_knowledge_for_format(self, node_list: List, component: str, round: int):
        """Update the knowledge pool for formats."""
        for p in node_list:
            fn = getattr(p, component.lower())[0]
            self.format_pool[component][fn]['chosen_count'] += 1
            self.format_pool[component][fn]['confidence_score'] += p.eval_score if p.eval_score is not None else 0

            knowledge = self.format_pool[component]
            log_total_count = math.log(sum(knowledge[k]['chosen_count'] for k in knowledge))
            exploration_weight = 0.001

            def uct(k):
                confidence_score = knowledge[k]['confidence_score']
                chosen_count = knowledge[k]['chosen_count']
                return confidence_score / (1 + chosen_count) + exploration_weight * math.sqrt(log_total_count / (1 + chosen_count))

            self.format_pool[component][fn]['uct_score'] = uct(fn)

    def update_knowledge(self, round: int):
        """Update the knowledge pool for all components."""
        for component in self.COMPONENT_KEYS:
            self._update_knowledge_for_format(
                self.prompt_history.get_modified_nodes(
                    self.prompt_history.get_nodes_by_round(round), component
                ), component, round
            )