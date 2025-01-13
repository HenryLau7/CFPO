from utils import get_component_desc

class BaseMutator:
    def __init__(self, mutation_llm, task, COMPONENT_KEYS):
        self.mutation_llm = mutation_llm
        self.task = task
        self.COMPONENT_KEYS = COMPONENT_KEYS
        self.component_desc = self.get_component_desc()
    
    def get_component_desc(self):
        descs = "\n".join([f"{i+1}. {item.upper()}: {get_component_desc(item)}" for i, item in enumerate(self.COMPONENT_KEYS)])
        return descs

    def _get_meta_prompt_header(self, prompt):

        return f"""
        I'm trying to write a prompt to {self.task.task_intention}.

        My current prompt comprises several essential segment, including:
        {self.component_desc}
        
        The whole prompt is:
        \"\"\"{str(prompt)}\"\"\"""".strip()