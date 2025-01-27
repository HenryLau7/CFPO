from .base import BaseMutator
import random
import re
from typing import List, Tuple, Optional
from utils import parse_tagged_text

class MonteCarloSampling(BaseMutator):
    def __init__(
        self,
        mutation_llm,
        task,
        COMPONENT_KEYS: List[str],
        logger=None,
    ):
        super().__init__(mutation_llm, task, COMPONENT_KEYS)
        self.logger = logger

    def __call__(self, prompt, num_prompt: int, num_component: int, round: int, temperature: float) -> List:
        """Generate new prompts by randomly mutating components."""
        new_prompts = self.generate_synonyms(prompt, num_prompt, num_component, round, temperature)
        return new_prompts

    def random_choose_component(self, n: int = 1) -> List[str]:
        """Randomly choose components to mutate."""
        return random.sample(self.COMPONENT_KEYS, random.randint(1, min(len(self.COMPONENT_KEYS), n)))

    def generate_synonyms(self, prompt, num_prompt: int = 3, num_component: int = 1, round: int = -1, temperature: float = 1) -> List:
        """Generate synonyms for a prompt by mutating selected components."""
        new_prompts = []

        for _ in range(num_prompt):
            selected_component_keys = self.random_choose_component(num_component)
            component_key_list, content_list = [], []

            for component_key in selected_component_keys:
                if component_key == "EXAMPLES":
                    content = self._generate_synonyms_for_examples(prompt, temperature=temperature)
                else:
                    content = self._generate_synonyms(prompt, component_key, temperature=temperature)

                component_key_list.append(component_key)
                content_list.append(content)

            new_prompt = prompt.generate(
                round=round,
                component_keys=component_key_list,
                component_contents=content_list,
                action_desc="random",
            )

            try:
                if str(new_prompt) and str(new_prompt) != str(prompt):
                    new_prompts.append(new_prompt)
            except Exception as e:
                self.logger.error(f"Error generating prompt with component keys {component_key_list}: {e}")
                continue

            # Log results
            self.logger.info(f"\n================ In Round {round} Generate synonyms prompt ================")
            for i, (key, content) in enumerate(zip(component_key_list, content_list)):
                self.logger.info(f"## Component KEY: {key}\n")
                self.logger.info(f"## New component: {str(content)}\n")
            self.logger.info(f"## Variation summary: at most {num_component}, actual variation {len(component_key_list)}\n")
            self.logger.info(f"## New prompt: {new_prompt}\n")

        return new_prompts

    def _generate_synonyms(self, prompt, component_name: str, temperature: float) -> str:
        """Generate synonyms for a specific component of the prompt."""
        prompt_to_generate_synonyms = f"""{self._get_meta_prompt_header(prompt)}

        Please create a different version of {component_name} segment without changing its semantic meaning. In case this segment is an empty string, generate a suitable one. The existing {component_name} segment contains:
        \"\"\"{getattr(prompt, component_name.lower())}\"\"\"

        The varied {component_name} segment is as follows:
        """
        prompt_to_generate_synonyms = '\n'.join([line.lstrip() for line in prompt_to_generate_synonyms.split('\n')])
        new_prompt_component = self.mutation_llm.inference(prompt_to_generate_synonyms, desc=f"get variation for {component_name}", temperature=temperature)
        new_prompt_component = re.sub(r'^[\n"\' ]+|[\n"\' ]+$', '', new_prompt_component)
        return prompt.prompt_renderer[1](new_prompt_component)

    def _generate_synonyms_for_examples(self, prompt, temperature: float) -> str:
        """Generate synonyms for the EXAMPLES section of the prompt."""
        prompt_to_generate_synonyms = f"""{self._get_meta_prompt_header(prompt)}

        The existing EXAMPLE set contains:
        \"\"\"{prompt.render_examples(prompt.examples)}\"\"\"

        Please generate a variation of the EXAMPLES set within the prompt while keeping the semantic meaning. The revision should represent ONLY ONE of the following specific actions:
        1. Addition: Incorporating one new example into the existing set.
        2. Deletion: Eliminating one single example from the current set.
        3. Modification: Changing the content of an example while maintaining its contextual relevance.
        Please present the results without indicating which action was taken. The varied EXAMPLES segment is as follows:
        """
        prompt_to_generate_synonyms = '\n'.join([line.lstrip() for line in prompt_to_generate_synonyms.split('\n')])
        new_prompt_component = self.mutation_llm.inference(prompt_to_generate_synonyms, desc="get variation for example", temperature=temperature)
        new_prompt_component = re.sub(r'^[\n"\' ]+|[\n"\' ]+$', '', new_prompt_component)
        return prompt.query_format[1](new_prompt_component, prompt.cot_hinter)

    def _parse_format(self, texts: List[str], format_name_pattern: str = r"<Format name: (?:\d+\.)?\s*(.*?)>\n<Description: (.*?)>\n(.+)") -> List[Tuple[str, str, str]]:
        """Parse text that is tagged with start and end tags."""
        outputs = []
        for t in texts:
            try:
                format_name, format_description, rendered_example = re.findall(format_name_pattern, t, re.DOTALL)[0]
                outputs.append((format_name, format_description, rendered_example))
            except Exception as e:
                self.logger.warning(f"Unexpected format encountered: {t}\nError: {e}")
        return outputs