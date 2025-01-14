import json

def json_renderer(
        task_instruction: str,
        task_detail: str,
        output_format: str,
        examples: str,
        query_part: str,
    ) -> str:
    """ Merge examples and query_part into Examples section and render as JSON
    """
    prompt = {
        "TaskInstruction": task_instruction.strip(),
        "TaskDetail": task_detail.strip(),
        "OutputFormat": output_format.strip(),
        "Examples": (examples.strip() + "\n\n" + query_part.strip()).strip()
    }
    return json.dumps(prompt, indent=4)

def json_extractor(prompt: str) -> str:
    """ Extract only the query_part section from a JSON prompt """
    try:
        prompt_dict = json.loads(prompt)
        examples = prompt_dict.get("Examples", "")
        # Extract the query_part by splitting from the last newline
        if examples:
            return examples.split("\n\n")[-1].strip()
        return ""
    except json.JSONDecodeError:
        return "Invalid JSON input"

if __name__ == '__main__':
    import json
    ## Example usage of json

    task_instruction = "Write a function that returns the sum of two numbers."
    task_detail = "The function should take two numbers as input and return their sum."
    output_format = "int: the sum of the two numbers"
    examples = "Example 1:\nInput: 1, 2\nOutput: 3\n\nExample 2:\nInput: -1, 1\nOutput: 0"
    query_part = "Input: 1, 2\nOutput:"

    # Render JSON prompt
    prompt = json_renderer(task_instruction, task_detail, output_format, examples, query_part)
    print("Generated JSON Prompt:")
    print(prompt)

    # Extract query part from JSON prompt
    extracted_query_part = json_extractor(prompt)
    print("\nExtracted Query Part:")
    print(extracted_query_part)
