from abc import ABC, abstractmethod

class BasePromptFormat(ABC):
    @abstractmethod
    def render(self, task_instruction: str, task_detail: str, output_format: str, examples: str) -> str:
        """Render the prompt in the specified format."""
        pass

    @abstractmethod
    def extract(self, response: str) -> str:
        """Extract the answer from the model's response."""
        pass


class MarkdownPromptFormat(BasePromptFormat):
    def render(self, task_instruction: str, task_detail: str, output_format: str, examples: str) -> str:
        return (
            f"### Task Instruction\n{task_instruction}\n\n"
            f"### Task Detail\n{task_detail}\n\n"
            f"### Output Format\n{output_format}\n\n"
            f"### Examples\n{examples}"
        )

    def extract(self, response: str) -> str:
        # Extract the answer from the response
        return response.split("### Answer")[-1].strip()