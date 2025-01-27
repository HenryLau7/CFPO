from .base import BaseMutator
from utils import parse_tagged_text, stringify_dict
import re
import math
import random
import inspect
from typing import Optional, Tuple, Callable, Dict, Any, List

class FormatMutator(BaseMutator):
    def __init__(
        self,
        mutation_llm,
        task,
        COMPONENT_KEYS: List[str],  # ['PROMPT_RENDERER', 'QUERY_FORMAT']
        prompt_history,
        search_pool: Dict,  # A dict: search_pool['prompt'], search_pool['query']
        select_method: str,
        logger=None,
    ):
        super().__init__(mutation_llm, task, COMPONENT_KEYS)
        self.task = task
        self.knowledge_components = COMPONENT_KEYS
        self.prompt_history = prompt_history
        self.format_pool = {'PROMPT_RENDERER': None, 'QUERY_FORMAT': None}
        self.search_pool = search_pool
        self.logger = logger
        self.select_method = select_method
        self._init_format_pool()

    def _init_format_pool(self):
        """Initialize the knowledge pool for formats."""
        self.format_pool["PROMPT_RENDERER"] = {
            fn[0]: {'confidence_score': 0, 'chosen_count': 0, 'uct_score': 0}
            for fn in self.search_pool["prompt"]
        }
        self.format_pool["QUERY_FORMAT"] = {
            fn[0]: {'confidence_score': 0, 'chosen_count': 0, 'uct_score': 0}
            for fn in self.search_pool["query"]
        }

    def __call__(self, prompts: List, num_select_formats: int, round: int) -> List:
        """Generate new prompts by mutating formats."""
        self.round = round

        if round == 2:
            new_prompts = []
            for prompt in prompts:
                new_prompts_per_prompt = [prompt]
                new_prompts_per_prompt += self.traverse_format(prompt, round)
                new_prompts.append(new_prompts_per_prompt)
            return new_prompts
        else:
            generated_prompt_renderer, generated_query_format = self.generate_new_format()
            selected_prompt_renderers, selected_query_formats = self.format_select(num_select_formats, round)

            prompt_renderers = [generated_prompt_renderer] + selected_prompt_renderers
            query_formats = [generated_query_format] + selected_query_formats

            new_prompts = self._apply_formats_to_prompts(prompts, prompt_renderers, query_formats, round)

        return new_prompts

    def traverse_format(self, prompt, round: int) -> List:
        """Traverse all formats and generate new prompts."""
        self.logger.info(f"\n================ In Round {round} Traverse all formats ================")
        current_prompt_renderer = prompt.prompt_renderer
        current_query_format = prompt.query_format
        new_prompts = []

        for prompt_renderer in self.search_pool['prompt']:
            for query_format in self.search_pool['query']:
                if prompt_renderer != current_prompt_renderer or query_format != current_query_format:
                    component_keys, component_contents = [], []
                    if prompt_renderer != current_prompt_renderer:
                        component_keys.append("PROMPT_RENDERER")
                        component_contents.append(prompt_renderer)
                    if query_format != current_query_format:
                        component_keys.append("QUERY_FORMAT")
                        component_contents.append(query_format)

                    new_prompt = prompt.generate(
                        round=round,
                        component_keys=component_keys,
                        component_contents=component_contents,
                        action_desc="traverse",
                    )
                    new_prompts.append(new_prompt)

        return new_prompts

    def _apply_formats_to_prompts(self, prompts: List, prompt_renderers: List, query_formats: List, round: int) -> List:
        """Apply formats to prompts and generate new prompts."""
        new_prompts = []

        def apply_format(prompt, format_type: str, format_func: Tuple[Callable, Callable], action_desc: str) -> Optional[str]:
            """Helper function to apply a format to a prompt and handle exceptions."""
            try:
                new_prompt = prompt.generate(
                    round=round,
                    component_keys=[format_type],
                    component_contents=[format_func],
                    action_desc=action_desc,
                )
                new_prompt.render_all()
                return str(new_prompt) if str(new_prompt) != str(prompt) else None
            except Exception as e:
                self.logger.error(f"Error generating prompt with {format_type.lower()}: {e}")
                return None

        for prompt in prompts:
            new_prompts_per_prompt = [prompt]

            # Apply the first prompt renderer and query format
            for format_type, formats, action_desc in [
                ("PROMPT_RENDERER", prompt_renderers, "generated_format"),
                ("QUERY_FORMAT", query_formats, "generated_format"),
            ]:
                result = apply_format(prompt, format_type, formats[0], action_desc)
                if result:
                    new_prompts_per_prompt.append(result)

            # Apply the remaining prompt renderers and query formats
            for format_type, formats, action_desc in [
                ("PROMPT_RENDERER", prompt_renderers[1:], "selected_format"),
                ("QUERY_FORMAT", query_formats[1:], "selected_format"),
            ]:
                for format_func in formats:
                    result = apply_format(prompt, format_type, format_func, action_desc)
                    if result:
                        new_prompts_per_prompt.append(result)

            new_prompts.append(new_prompts_per_prompt)

        return new_prompts

    def generate_new_format(self) -> Optional[Tuple]:
        def generate_format(generate_func: Callable, generate_code_func: Callable, search_pool_key: str, format_pool_key: str, **kwargs) -> Optional[Tuple[Callable, Callable]]:
            generated_format = generate_func()
            if not generated_format:
                return None

            generated_code = generate_code_func(generated_format, self.search_pool[search_pool_key], self.search_pool[f"{search_pool_key}_desc"], **kwargs)
            if not generated_code:
                return None

            name, description, render_code, extractor_code = generated_code
            namespace: Dict[str, Any] = {}

            try:
                exec(render_code, globals(), namespace)
                exec(extractor_code, globals(), namespace)
            except Exception as e:
                self.logger.error(f"Error executing generated code: {e}")
                return None

            import pdb;pdb.set_trace()
            renderer_func = namespace[f"{name}_renderer"]
            extractor_func = namespace[f"{name}_extractor"]

            if callable(renderer_func) and callable(extractor_func):
                self.search_pool[search_pool_key].append((renderer_func, extractor_func))
                self.search_pool[f"{search_pool_key}_desc"][renderer_func] = description
                self.format_pool[format_pool_key][renderer_func] = {'confidence_score': 0, 'chosen_count': 0, 'uct_score': 0}
                return (renderer_func, extractor_func)
            return None

        generated_prompt_renderer = generate_format(
            self._generate_prompt_renderer,
            self._generate_prompt_renderer_code,
            'prompt',
            'PROMPT_RENDERER',
            temperature=1
        )

        generated_query_format = generate_format(
            self._generate_query_format,
            self._generate_query_format_code,
            'query',
            'QUERY_FORMAT',
            temperature=1
        )

        return (generated_prompt_renderer, generated_query_format)

    def format_select(self, num_prompt: int, round: int) -> Tuple[List, List]:
        """Apply knowledge-based formats."""

        if self.select_method == "UCT":
            new_prompt_renderers = sorted(
                self.format_pool["PROMPT_RENDERER"],
                key=lambda k: self.format_pool["PROMPT_RENDERER"][k]['uct_score'],
                reverse=True
            )[:num_prompt]
            new_query_formats = sorted(
                self.format_pool["QUERY_FORMAT"].keys(),
                key=lambda k: self.format_pool["QUERY_FORMAT"][k]['uct_score'],
                reverse=True
            )[:num_prompt]

            new_prompt_renderers = [(k, v) for (k, v) in  self.search_pool['prompt'] if k in new_prompt_renderers]
            new_query_formats = [(k, v) for (k, v) in self.search_pool['query'] if k in new_query_formats]

        elif self.select_method == "Random":
            new_prompt_renderers = random.sample(self.search_pool['prompt'], num_prompt)
            new_query_formats = random.sample(self.search_pool['query'], num_prompt)
        else:
            raise NotImplementedError

        self.logger.info(f"Selected formats:\nPrompt renderers: {new_prompt_renderers}\nQuery formats: {new_query_formats}")
        return (new_prompt_renderers, new_query_formats)

    def _generate_prompt_renderer(self) -> Optional[Tuple]:
        """Generate a new PROMPT_RENDERER."""
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

        The whole prompt is  \"\"\"{str(self.prompt_history.beam_history[self.round-1][0]).strip()}\"\"\"

        We have some preset PROMPT_RENDERER candidates, here are our whole search pool:
        {format_fn_desc_string}
        
        Here are two examples from our PROMPT_RENDERER candidates as for your reference:
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

        Please generate ONE new format for the PROMPT_RENDERER segment, its description and render the {{TASK_INSTRUCTION}}, {{TASK_DETAIL}}, {{OUTPUT_FORMAT}} and {{EXAMPLES}} segments using this new format. The new format could either be distinct from the existing formats, or a variation of an existing format. 
        
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
        response = self.mutation_llm.inference(prompt_to_generate_format, desc="generate prompt format", temperature=1)
        # self.logger.info(f"\n### Prompt to generate prompt renderer: {prompt_to_generate_format}")

        new_formats = self._parse_format(parse_tagged_text(response, "<START>", "<END>"))

        if len(new_formats) == 0:
            return None
        return new_formats[0]

    def _generate_prompt_renderer_code(self, new_format, search_pool, format_desc, temperature):
        """
        new_format: (format_name, format_description, rendered_example)
        return: (name, description, render_new_code, extractor_new_code)
        """
        (format_name, format_description, rendered_example) = new_format

        format_fn_desc = []
        for key, content in format_desc.items():
            format_fn_desc.append((key.__name__[:-9], content))
        format_fn_desc_string = "\n".join([f"{name}: {desc}" for name,desc in format_fn_desc])

        prompt_to_generate_code = f"""{self._get_meta_prompt_header()}

        We have some preset PROMPT_RENDERER candidates, here are our whole search pool:
        {format_fn_desc_string}

        Here are two code implementations from our PROMPT_RENDERER candidates as for your reference:
        <Format name: markdown>
        <Renderer code>
        {inspect.getsource(search_pool[1][0])}
        <Extractor code>
        {inspect.getsource(search_pool[1][1])}

        <Format name: xml>
        <Renderer code>
        {inspect.getsource(search_pool[4][0])}
        <Extractor code>
        {inspect.getsource(search_pool[4][1])}

        Here is a example rendered by a new format:
        {rendered_example}

        Please generate the code for this provided example based on the new PROMPT_RENDERER. Ensure that both the renderer and extractor functions are included. The generated code should be plain Python code without any Markdown syntax or language identifiers such as ```python or '''python. Please output the code directly without any additional formatting. Note that the name of the functions must be completely SAME with the name of format, i.e. [Format name]_renderer and [Format_name]_extractor.
        Please encapsulate the using the following format:
        
        <START>
        <Format name: {format_name}>
        <Description: {format_description}>
        <Renderer code>
        [Renderer code]
        <Extractor code>
        [Extractor code]
        <END>
        """

        prompt_to_generate_code = '\n'.join([line.lstrip() for line in prompt_to_generate_code.split('\n')])
        response = self.mutation_llm.inference(prompt_to_generate_code, desc="generate prompt format code", temperature=temperature)
        # self.logger.info(f"\n### Prompt to generate prompt renderer code: {prompt_to_generate_code}")

        new_formats = self._parse_format_code(parse_tagged_text(response, "<START>", "<END>"))
        if len(new_formats) == 0:
            return None
        return new_formats[0]    

    def _generate_query_format_code(self, new_format, search_pool, format_desc, temperature):
        """
        new_format: (format_name, format_description, rendered_example)
        return: (name, description, render_new_code, extractor_new_code)
        """
        (format_name, format_description, rendered_example) = new_format

        format_fn_desc = []
        for key, content in format_desc.items():
            format_fn_desc.append((key.__name__[:-9], content))
        format_fn_desc_string = "\n".join([f"{name}: {desc}" for name,desc in format_fn_desc])

        if self.task.__class__.__name__ in ['GSM8KTask', 'MATHTask']:
            example = {
                "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is: 6."
            }
            format_name_exs = ['QA', 'IR']
            task_specific_instruction = "Ensure the format is suitable for mathematical problem-solving and includes clear instructions for step-by-step reasoning."
            rendered_example_1 = self.search_pool['query'][1][0](example["question"], example["answer"], self.prompt_history.beam_history[self.round-1][0].cot_hinter)
            rendered_example_2 = self.search_pool['query'][2][0](example["question"], example["answer"], self.prompt_history.beam_history[self.round-1][0].cot_hinter)

        elif self.task.__class__.__name__ in ['MultipleChoiceTask']:
            example = {
                "question": "Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.",
                "choices": ["True, True",
                            "False, False",
                            "True, False",
                            "False, True"],
                "answer": "A cyclic group is a group that is generated by a single element. Hence a subgroup generated by a single element of a group is cyclic and Statement 1 is True. The answer is (C)."
            }
            format_name_exs = ['Plain', 'Markdown']
            task_specific_instruction = "Ensure the format is clear, structured, and aligned with commonly used query formats."
            rendered_example_1 = self.search_pool['query'][0][0](example["question"], example['choices'], example["answer"], self.prompt_history.beam_history[self.round-1][0].cot_hinter).strip()
            rendered_example_2 = self.search_pool['query'][1][0](example["question"], example['choices'], example["answer"], self.prompt_history.beam_history[self.round-1][0].cot_hinter).strip()

        prompt_to_generate_code = f"""{self._get_meta_prompt_header()}

        We have some preset QUERY_FORMAT candidates, here are our whole search pool:
        {format_fn_desc_string}

        Here are two code implementations from our QUERY_FORMAT candidates as for your reference:
        <Format name: {format_name_exs[0]}>
        <Renderer code>
        {inspect.getsource(search_pool[0][0])}
        <Extractor code>
        {inspect.getsource(search_pool[0][1])}

        <Format name: {format_name_exs[1]}>
        <Renderer code>
        {inspect.getsource(search_pool[3][0])}
        <Extractor code>
        {inspect.getsource(search_pool[3][1])}

        Here is the example rendered by the new format:
        {rendered_example}

        Please generate the code for this provided example based on the new QUERY_FORMAT. Ensure that both the renderer and extractor functions are included. The generated code should be plain Python code without any Markdown syntax or language identifiers such as ```python or '''python. Please output the code directly without any additional formatting.

        Please encapsulate the using the following format:
        
        <START>
        <Format name: {format_name}>
        <Description: {format_description}>
        <Renderer code>
        [Renderer code]
        <Extractor code>
        [Extractor code]
        <END>
        """

        prompt_to_generate_code = '\n'.join([line.lstrip() for line in prompt_to_generate_code.split('\n')])
        response = self.mutation_llm.inference(prompt_to_generate_code, desc="generate query format code", temperature=temperature)

        new_formats = self._parse_format_code(parse_tagged_text(response, "<START>", "<END>"))
        # self.logger.info(f"\n### Prompt to generate query format code: {prompt_to_generate_code}")

        if len(new_formats) == 0:
            return None
        return new_formats[0]

    def _generate_query_format(self) -> Optional[Tuple]:
        """Generate a new QUERY_FORMAT."""
        if self.task.__class__.__name__ in ['GSM8KTask', 'MATHTask']:
            example = {
                "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is: 6."
            }
            format_name = ['QA', 'IR']
            task_specific_instruction = "Ensure the format is suitable for mathematical problem-solving and includes clear instructions for step-by-step reasoning."
            rendered_example_1 = self.search_pool['query'][1][0](example["question"], example["answer"], self.prompt_history.beam_history[self.round-1][0].cot_hinter)
            rendered_example_2 = self.search_pool['query'][2][0](example["question"], example["answer"], self.prompt_history.beam_history[self.round-1][0].cot_hinter)

        elif self.task.__class__.__name__ in ['MultipleChoiceTask']:
            example = {
                "question": "Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.",
                "choices": ["True, True",
                            "False, False",
                            "True, False",
                            "False, True"],
                "answer": "A cyclic group is a group that is generated by a single element. Hence a subgroup generated by a single element of a group is cyclic and Statement 1 is True. The answer is (C)."
            }
            format_name = ['plain', 'markdown']
            task_specific_instruction = "Ensure the format is clear, structured, and aligned with commonly used query formats."
            rendered_example_1 = self.search_pool['query'][0][0](example["question"], example['choices'], example["answer"], self.prompt_history.beam_history[self.round-1][0].cot_hinter).strip()
            rendered_example_2 = self.search_pool['query'][1][0](example["question"], example['choices'], example["answer"], self.prompt_history.beam_history[self.round-1][0].cot_hinter).strip()

        format_fn_desc = []
        for key, content in self.search_pool['query_desc'].items():
            format_fn_desc.append((key.__name__[:-9], content))
        format_fn_desc_string = "\n".join([f"{name}: {desc}" for name, desc in format_fn_desc])

        prompt_to_generate_format = f"""{self._get_meta_prompt_header()}

        The whole prompt is  \"\"\"{str(self.prompt_history.beam_history[self.round-1][0]).strip()}\"\"\"

        We have some preset QUERY_FORMAT candidates, here are our whole search pool:
        {format_fn_desc_string}
        
        Here are two examples from our QUERY_FORMAT candidates as for your reference:
        <Format name: {format_name[0]}>
        {rendered_example_1}

        <Format name: {format_name[1]}>
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
        response = self.mutation_llm.inference(prompt_to_generate_format, desc="generate query format", temperature=1)
        # self.logger.info(f"\n### Prompt to generate query format: {prompt_to_generate_format}")

        new_formats = self._parse_format(parse_tagged_text(response, "<START>", "<END>"))

        if len(new_formats) == 0:
            return None
        return new_formats[0]

    def _get_meta_prompt_header(self) -> str:
        """Get the meta prompt header for format generation."""
        return f"""I'm trying to write a prompt to {self.task.task_intention}.
        The ultimate aim is to create a prompt that is clear, structured, and efficient, leading to accurate responses from the AI model. The structure of the prompt includes several essential elements: {self.component_desc}""".strip()

    def _update_format_pool(self, node_list: List, component: str, round: int):
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

    def update_format_pool(self, round: int):
        for format in ['PROMPT_RENDERER', 'QUERY_FORMAT']:
            self._update_format_pool(
                self.prompt_history.get_modified_nodes(
                    self.prompt_history.get_nodes_by_round(round), format
                ), format, round
            )

    def _parse_format(self, texts, format_name_pattern=r"<Format name: (?:\d+\.)?\s*(.*?)>\n<Description: (.*?)>\n(.+)"):
        """ Parse text that is tagged with start and end tags."""
        outputs = []
        for t in texts:
            try:
                format_name, format_description, rendered_example = re.findall(format_name_pattern, t, re.DOTALL)[0]
                outputs.append((format_name, format_description, rendered_example))
            except:
                self.logger.error(f"Error parsing format: {t}")
                pass
        return outputs
   
    def _parse_format_code(
        self, 
        texts, 
        format_code_pattern=r"<Format name:\s*(.*?)>\s*<Description:\s*(.*?)>\s*<Renderer code>\s*(.*?)\s*<Extractor code>\s*(.*?)$"):
        """ Parse text that is tagged with start and end tags."""
        outputs = []
        for t in texts:
            try:
                format_name, format_description, renderer_code, extractor_code = re.findall(format_code_pattern, t, re.DOTALL)[0]
                if renderer_code.startswith("```") and renderer_code.endswith("```"):
                    renderer_code = renderer_code.strip("```").strip()
                if extractor_code.startswith("```") and extractor_code.endswith("```"):
                    extractor_code = extractor_code.strip("```").strip()
                outputs.append((format_name, format_description, renderer_code, extractor_code))
            except:
                self.logger.error(f"Error parsing format code: {t}")
                continue
        return outputs