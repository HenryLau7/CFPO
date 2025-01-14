
def latex_renderer(
        task_instruction: str,
        task_detail: str,
        output_format: str,
        examples: str,
        query_part: str,
    ) -> str:
    """ Render all components in LaTeX format with \section{component-key} """
    from collections import OrderedDict
    prompt = OrderedDict([
        ("TaskInstruction", task_instruction),
        ("TaskDetail", task_detail),
        ("OutputFormat", output_format),
        ("Examples", examples.strip() + "\n\n" + query_part.strip())
    ])
    latex_content = ""
    for name, content in prompt.items():
        if content and len(content) > 0:
            latex_content += f"\section{{{name}}}\n{content.strip()}\n\n"
    return latex_content.strip()



# def latex_extractor(component: str) -> str:
#     """
#     Extract content from a given LaTeX formatted component.
#     """
#     import re
#     pattern = re.compile(r'\section\{.*?\}\n(.*)', re.DOTALL)
#     match = pattern.match(component)
#     if match:
#         return match.group(1).strip()
#     else:
#         return component.strip()
def latex_extractor(content: str) -> str:
    # Remove LaTeX commands like \section{{component key}}
    import re
    cleaned_text = re.sub(r'\\section\{[^}]*\}', '', content)
    return cleaned_text.strip()


if __name__ == '__main__':
    task_instruction = "Write a function that returns the sum of two numbers."
    task_detail = "The function should take two numbers as input and return their sum."
    output_format = "int: the sum of the two numbers"
    examples = "Example 1:\nInput: 1, 2\nOutput: 3\n\nExample 2:\nInput: -1, 1\nOutput: 0"
    query_part = "Input: 1, 2\nOutput:"

    prompt = latex_renderer(task_instruction, task_detail, output_format, examples, query_part)
    print(prompt)
    extracted_query_part = latex_extractor(prompt)
    print(extracted_query_part)  # Expected: "Input: 1, 2\nOutput:"