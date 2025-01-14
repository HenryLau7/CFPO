def direct_joint_renderer(task_instruction: str,
           task_detail: str,
           output_format: str,
           examples: str,
           query_part: str,
           ):
    prompt = ""
    for item in [task_instruction, task_detail, output_format, examples, query_part]:
        if item and len(item) > 0:
            prompt += item.strip() + "\n\n"
    return prompt.strip()

def direct_joint_extractor(prompt: str):
    return prompt