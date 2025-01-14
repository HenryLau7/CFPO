import re
from collections import OrderedDict

def plain_renderer(
        task_instruction: str,
        task_detail: str,
        output_format: str,
        examples: str,
        query_part: str,
    ) -> str:
    prompt = OrderedDict([
        ("Task Instruction", task_instruction),
        ("Task Detail", task_detail),
        ("Output Format", output_format),
        ("Examples", examples.strip() + "\n\n" + query_part.strip())
    ])
    formatted_prompt = ""
    for name, content in prompt.items():
        if content and len(content) > 0:
            formatted_prompt += f"{name}:\n{content.strip()}\n\n"
    
    return formatted_prompt.strip()
    
def plain_extractor(component: str) -> str:
    import re
    pattern = re.compile(r'^(.*?):\n(.*)', re.DOTALL)
    match = pattern.match(component)
    if match:
        return match.group(2).strip()
    else:
        return component.strip()

if __name__ == '__main__':
    task_instruction = "Write a function that returns the sum of two numbers."
    task_detail = "The function should take two numbers as input and return their sum."
    output_format = "int: the sum of the two numbers"
    examples = "Example 1:\nInput: 1, 2\nOutput: 3\n\nExample 2:\nInput: -1, 1\nOutput: 0"
    query_part = "Input: 1, 2\nOutput:"

    prompt = plain_renderer(task_instruction, task_detail, output_format, examples, query_part)
    print(prompt)
    extracted_query_part = plain_extractor(prompt)
    print(extracted_query_part)