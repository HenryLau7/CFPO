def markdown_renderer(
        task_instruction: str,
        task_detail: str,
        output_format: str,
        examples: str,
        query_part: str,
    ):
    """ Merge examples and query_part into Examples section
    """
    prompt = ""
    from collections import OrderedDict
    section_name_dic = OrderedDict([
        ("Task Instruction", task_instruction), 
        ("Task Detail", task_detail),
        ("Output Format", output_format),
        ("Examples", examples.strip() + "\n\n" + query_part.strip()),
    ])
    for name, part in section_name_dic.items():
        if part and len(part) > 0:
            prompt += f"##### {name}\n" + part.strip() + "\n\n"
    return prompt.strip()

def markdown_extractor(prompt):
    pattern = r'^#+ .*?\n(.*)'
    import re
    match = re.search(pattern, prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return prompt.strip()

if __name__ == '__main__':
    task_instruction = "Write a function that returns the sum of two numbers."
    task_detail = "The function should take two numbers as input and return their sum."
    output_format = "int: the sum of the two numbers"
    examples = "Example 1:\nInput: 1, 2\nOutput: 3\n\nExample 2:\nInput: -1, 1\nOutput: 0"
    query_part = "What is the sum of 3 and 4?"

    prompt = markdown_renderer(task_instruction, task_detail, output_format, examples, query_part)
    print(prompt)

    extracted_query_part = markdown_extractor(prompt)
    print(extracted_query_part) 