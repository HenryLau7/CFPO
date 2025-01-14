def xml_renderer(
        task_instruction: str,
        task_detail: str,
        output_format: str,
        examples: str,
        query_part: str,
    ) -> str:
    """ Merge examples and query_part into Examples section and render as XML
    """
    from collections import OrderedDict
    import xml.etree.ElementTree as ET
    prompt = OrderedDict([
        ("TaskInstruction", task_instruction),
        ("TaskDetail", task_detail),
        ("OutputFormat", output_format),
        ("Examples", examples.strip() + "\n\n" + query_part.strip())
    ])
    
    xml_parts = []
    for name, part in prompt.items():
        if part and len(part) > 0:
            element = ET.Element(name)
            element.text = part.strip()
            xml_str = ET.tostring(element, encoding='unicode', method='xml')
            if name == "Examples":
                xml_str = xml_str.replace(f"</{name}>", "")  # Remove closing tag for Examples
            xml_parts.append(xml_str)
    return "\n".join(xml_parts)


def xml_extractor(prompt: str) -> str:
    # Remove all XML tags, including incomplete ones
    import re
    cleaned_text = re.sub(r'<[^>]*>', '', prompt)
    return cleaned_text.strip()


if __name__ == '__main__':
    task_instruction = "Write a function that returns the sum of two numbers."
    task_detail = "The function should take two numbers as input and return their sum."
    output_format = "int: the sum of the two numbers"
    examples = "Example 1:\nInput: 1, 2\nOutput: 3\n\nExample 2:\nInput: -1, 1\nOutput: 0"
    query_part = "Input: 1, 2\nOutput:"

    prompt = xml_renderer(task_instruction, task_detail, output_format, examples, query_part)
    print(prompt)
    extracted_query_part = xml_extractor('<OutputFormat>int: the sum of the two numbers</OutputFormat>')
    print(extracted_query_part)