import os
import pickle
from typing import List, Dict, Any, Optional, Callable, Tuple

class Prompt:
    """
    Represents a prompt in a hierarchical structure, allowing for the generation of child prompts
    with modifications to specific components.
    """

    def __init__(
        self,
        task: str,
        round: int,
        query_part: str,
        task_instruction: str,
        task_detail: str,
        output_format: str,
        example_hinter: str,
        examples: List[Any],
        prompt_renderer_fn: Callable,
        prompt_extract_fn: Callable,
        query_renderer_fn: Callable,
        query_extract_fn: Callable,
        action_desc: str = 'initialization',
        action_detail: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        """
        Initializes a new Prompt instance.

        Args:
            round (int): The round number of the prompt.
            query_part (str): The query part of the prompt.
            task_instruction (str): The task instruction.
            task_detail (str): Detailed description of the task.
            output_format (str): The format of the output.
            example_hinter (str): The hint for examples.
            examples (List[Any]): List of examples.
            prompt_renderer_fn (Callable): Function to render the prompt.
            prompt_extract_fn (Callable): Function to extract information from the prompt.
            query_renderer_fn (Callable): Function to format the query.
            query_extract_fn (Callable): Function to extract information from the query.
            action_desc (str, optional): Description of the action. Defaults to 'initialization'.
            action_detail (Optional[Dict[str, Any]], optional): Detailed information about the action. Defaults to None.
        """
        self.parent = None
        self.children = []
        self.eval_score = None
        self.test_score = None
        self.improved_score = None
        self.round = round
        self.query_part = query_part
        self.task_instruction = task_instruction
        self.task_detail = task_detail
        self.output_format = output_format
        self.example_hinter = example_hinter
        self.examples = examples
        self.prompt_renderer = (prompt_renderer_fn, prompt_extract_fn)
        self.query_format = (query_renderer_fn, query_extract_fn)
        self.action_desc = action_desc
        self.action_detail = action_detail
        self.task = task
        self.cot_hinter = kwargs.get('cot_hinter', 'Let\'s think step by step.') if task in ['GSM8K', 'MATH'] else None
    
    def render_examples(self, examples: List[Dict[str, Any]]) -> str:
        """
        Renders the examples into a string.

        Args:
            examples (List[Dict[str, Any]]): List of examples to render.

        Returns:
            str: Rendered examples as a string.
        """
        example_str = self.example_hinter + '\n'
        query_renderer = self.query_format[0]


        if self.task in ['MultipleChoice']:
            example_str += "\n\n".join([query_renderer(question = example["question"], choices = example['choices'], cot_hinter = self.cot_hinter, answer = example["answer"]) for example in examples])
        else:
            example_str += "\n\n".join(
            query_renderer(question=example["question"], answer=example["answer"], cot_hinter=self.cot_hinter)
            for example in examples
        )
        return example_str

    def render_one_example(self, example: Dict[str, Any]) -> str:
        """
        Renders a single example into a string.

        Args:
            example (Dict[str, Any]): The example to render.

        Returns:
            str: Rendered example as a string.
        """
        query_renderer = self.query_format[0]

        if self.task in ['MultipleChoice']:
            return query_renderer(question = example["question"], choices = example['choices'], cot_hinter = self.cot_hinter, answer = example["answer"])
        else:
            return query_renderer(question=example["question"], answer=example["answer"], cot_hinter=self.cot_hinter)
       
    def render_query(self, question: str, **kwargs) -> str:
        """
        Renders a query into a string.

        Args:
            question (str): The question to render.

        Returns:
            str: Rendered query as a string.
        """
        query_renderer = self.query_format[0]

        if self.task in ['MultipleChoice']:
            choices = kwargs['choices']
            return query_renderer(question = question, choices = choices, cot_hinter=self.cot_hinter, answer = '')
        else:
            return query_renderer(question=question, cot_hinter=self.cot_hinter, answer='')
       
    def render_all(self) -> str:
        """
        Renders the entire prompt into a string.

        Returns:
            str: Rendered prompt as a string.
        """
        prompt_renderer = self.prompt_renderer[0]
        return prompt_renderer(
            task_instruction=self.task_instruction,
            task_detail=self.task_detail,
            output_format=self.output_format,
            examples=self.render_examples(self.examples),
            query_part=self.query_part
        )
    
    def __str__(self) -> str:
        return self.render_all()

    def generate(
        self,
        round: int,
        component_keys: List[str],
        component_contents: List[Any],
        action_desc: str
    ) -> 'Prompt':
        """
        Generates a child prompt with modifications to specific components.

        Args:
            round (int): The round number of the child prompt.
            component_keys (List[str]): List of component keys to modify.
            component_contents (List[Any]): List of new values for the components.
            action_desc (str): Description of the action.

        Returns:
            Prompt: The generated child prompt.
        """
        child_prompt = Prompt(
            round=round,
            task=self.task,
            query_part=self.query_part,
            task_instruction=self.task_instruction,
            task_detail=self.task_detail,
            output_format=self.output_format,
            example_hinter=self.example_hinter,
            examples=self.examples,
            prompt_renderer_fn=self.prompt_renderer[0],
            prompt_extract_fn=self.prompt_renderer[1],
            query_renderer_fn=self.query_format[0],
            query_extract_fn=self.query_format[1],
            action_desc=action_desc
        )

        child_prompt.parent = self
        self.children.append(child_prompt)

        action_detail_temp = {}
        component_pair = {}

        for key, content in zip(component_keys, component_contents):
            if content is None:
                continue
            if key in ['EXAMPLES', 'QUERY_FORMAT', 'PROMPT_RENDERER']:
                component_pair[key.lower()] = content
            else:
                content = str(content).strip('\'"')
                component_pair[key.lower()] = content.strip('\'"')

        for key, content in component_pair.items():
            action_detail_temp[key] = {
                'original': getattr(self, key),
                'new': content
            } 
            setattr(child_prompt, key, content)
        
        child_prompt.action_detail = action_detail_temp

        return child_prompt


class PromptHistory:
    def __init__(self, root_path, init_prompt: Prompt, init_round: int = 0):
        self.root_path = root_path
        self.root = init_prompt
        self.round = init_round
        self.format_pool = {}
        self.beam_history = {}

    def save(self, path: str) -> None:
        output_path = os.path.join(self.root_path, path)
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"{self.round}.pkl")
        pickle.dump(self, open(output_path, "wb"))
    @classmethod
    def load(cls, project_name: str, round: int) -> 'PromptHistory':
        path = os.path.join(self.root_path, project_name, f"{round}.pkl")
        return pickle.load(open(path, "rb"))

    def add_root(self, prompt: Prompt) -> None:
        if self.root is None:
            self.root = prompt
        else:
            raise ValueError("Root already exists")

    def get_nodes_by_round(self, round: int) -> List[Prompt]:
        nodes = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if node.round == round:
                nodes.append(node)
            queue.extend(node.children)
        return nodes

    def all_nodes(self) -> List[Prompt]:
        nodes = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue.extend(node.children)
        return nodes

    def get_best_and_worst_nodes(self, node_list: List[Prompt], n: int) -> Tuple[List[Prompt], List[Prompt]]:
        for node in node_list:
            if node.improved_score is None:
                try:
                    node.improved_score = node.eval_score - node.parent.eval_score
                except:
                    node.improved_score = 0
        sorted_nodes = sorted(node_list, key=lambda node: node.improved_score)
        n1, n2 = (len(sorted_nodes) // 2, len(sorted_nodes) - len(sorted_nodes) // 2) if len(sorted_nodes) < 2 * n else (n, n)
        best_nodes = [node for node in sorted_nodes[-n1:] if node.improved_score > 0][::-1]
        worst_nodes = [node for node in sorted_nodes[:n2] if node.improved_score < 0]
        return best_nodes, worst_nodes

    def get_modified_nodes(self, node_list: List[Prompt], component_key: str) -> List[Prompt]:
        return [node for node in node_list if node.parent and getattr(node, component_key.lower()) != getattr(node.parent, component_key.lower())]

    def get_history(self, node: Prompt) -> List[Prompt]:
        history = []
        while node:
            history.append(node)
            node = node.parent
        return history

if __name__ == "__main__":
    prompt = Prompt(
        round=0,
        task = 'commensense',
        query_part="What is the capital of France?",
        task_instruction="Answer the following question.",
        task_detail="Provide the correct answer to the question.",
        output_format="The capital of France is [answer].",
        example_hinter="Here are some examples:",
        examples=[{"question": "What is the capital of Germany?", "answer": "Berlin"}],
        prompt_renderer_fn=lambda task_instruction, task_detail, output_format, examples, query_part: f"{task_instruction}\n{task_detail}\n{output_format}\n{examples}\n{query_part}",
        prompt_extract_fn=lambda x: x,
        query_renderer_fn=lambda question, answer, cot_hinter: f"Q: {question}\nA: {answer}",
        query_extract_fn=lambda x: x
    )
    print(str(prompt))