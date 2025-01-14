def html_renderer(
        task_instruction: str,
        task_detail: str,
        output_format: str,
        examples: str,
        query_part: str,
    ) -> str:
    """ Merge examples and query_part into Examples section and render as HTML
    """
    from collections import OrderedDict
    prompt = OrderedDict([
        ("TaskInstruction", task_instruction),
        ("TaskDetail", task_detail),
        ("OutputFormat", output_format),
        ("Examples", examples.strip() + "\n\n" + query_part.strip())
    ])
    
    html_content = ""
    
    for name, part in prompt.items():
        if part and len(part) > 0:
            if name == "Examples":
                html_content += f"<div class='{name}'>\n  <h2>{name}</h2>\n  <p>{part.strip()}"
            else:
                html_content += f"<div class='{name}'>\n  <h2>{name}</h2>\n  <p>{part.strip()}</p>\n</div>\n"
    return html_content


def html_extractor(component: str) -> str:
    """
    Extract content from a given HTML component.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(component, 'html.parser')
    return soup.get_text().strip()

if __name__ == '__main__':
    task_instruction = "Write a function that returns the sum of two numbers."
    task_detail = "The function should take two numbers as input and return their sum."
    output_format = "int: the sum of the two numbers"
    examples = "Example 1:\nInput: 1, 2\nOutput: 3\n\nExample 2:\nInput: -1, 1\nOutput: 0"
    query_part = "Input: 1, 2\nOutput:"

    prompt = html_renderer(task_instruction, task_detail, output_format, examples, query_part)
    print(prompt)
    extracted_query_part = html_extractor(prompt)
    print(extracted_query_part)