import re
def concise_bullet_points_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    prompt_items = [
        ("Task Instruction", task_instruction),
        ("Task Detail", task_detail),
        ("Output Format", output_format),
        ("Examples", examples),
        ("Query", query_part)
    ]
    prompt = ""
    for title, content in prompt_items:
        if content and len(content) > 0:
            prompt += f"- {title}: {content}\n\n"
    return prompt.strip()

def concise_bullet_points_extractor(prompt: str) -> str:
    
    cleaned_text = re.sub(r'- [^:]+: ', '', prompt)
    return cleaned_text.strip()

def structured_summary_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    sections = [("Task Instruction", task_instruction), ("Task Detail", task_detail), ("Output Format", output_format), ("Examples", examples + "\n\n" + query_part)]
    rendered_output = ""
    for name, content in sections:
        if content:
            rendered_output += f"- {name}:\n"
            for line in content.strip().split("\n"):
                rendered_output += f"  - {line}\n"
    return rendered_output.strip()

def structured_summary_extractor(prompt: str) -> str:
    
    cleaned_text = re.sub(r'- [^:]+:\n\s*- ', '', prompt)
    cleaned_text = re.sub(r'- [^:]+:\n', '', cleaned_text)
    return cleaned_text.strip()

def interactive_dialogue_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    dialogue = ""
    components = [
        ("User: Can you explain the task instruction for a mathematical challenge?", task_instruction),
        ("User: What should I keep in mind while dealing with task details?", task_detail),
        ("User: How should the output be formatted?", output_format),
        ("User: Can you show me some examples?", examples),
        (query_part, "")
    ]    
    for user_prompt, ai_response in components:
        if ai_response:
            dialogue += f"{user_prompt}\nAI: {ai_response.strip()}\n\n"
        else:
            dialogue += f"{user_prompt}\n\n"
    return dialogue.strip()

def interactive_dialogue_extractor(prompt: str) -> str:
    
    cleaned_text = re.sub(r'User: [^\n]+\nAI: ', '', prompt)
    cleaned_text = re.sub(r'User: [^\n]+\n', '', cleaned_text)
    return cleaned_text.strip()

def simplified_markdown_bullet_points_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    prompt = ""
    for title, content in [("Task Instruction", task_instruction), ("Task Detail", task_detail), ("Output Format", output_format), ("Examples", examples), ("Query", query_part)]:
        if content and len(content) > 0:
            prompt += f"- **{title}**\n"
            for line in content.strip().split("\n"):
                if line:
                    prompt += f"- {line}\n"
            prompt += "\n"
    return prompt.strip()

def simplified_markdown_bullet_points_extractor(prompt: str) -> str:
    
    cleaned_text = re.sub(r'- \*\*[^*]+\*\*\n', '', prompt)
    cleaned_text = re.sub(r'- ', '', cleaned_text)
    return cleaned_text.strip()

def tabular_sections_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    sections = [("Task Instruction", task_instruction), ("Task Detail", task_detail), ("Output Format", output_format), ("Examples", examples), ("Query", query_part)]
    rendered_table = ""
    for section_name, content in sections:
        if content:
            rendered_table += f"| {section_name} | {content.strip()} |\n"
    return rendered_table

def tabular_sections_extractor(prompt: str) -> str:
    
    cleaned_text = re.sub(r'\| [^|]+ \| ', '', prompt)
    cleaned_text = re.sub(r' \|', '', cleaned_text)
    return cleaned_text.strip()


def descriptive_subheadings_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    prompt = ""
    segments = [("Understanding the Task: A Foundation for Mathematical Problem-Solving", task_instruction),
                ("Decoding Mathematical Language in Real-World Scenarios", task_detail),
                ("Ensuring Your Solution Fits the Scenario Perfectly", output_format),
                ("Examples to Illuminate the Path", examples),
                ("Query", query_part)]                
    for title, content in segments:
        if content and len(content) > 0:
            prompt += f"**{title}**\n{content.strip()}\n\n"
    return prompt.strip()

def descriptive_subheadings_extractor(prompt: str) -> str:
    
    cleaned_text = re.sub(r'\*\*[^*]+\*\*\n', '', prompt)
    return cleaned_text.strip()

def checklist_format_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    prompt = ""
    checklist_items = [
        ("Task Instruction", task_instruction),
        ("Task Detail", task_detail),
        ("Output Format", output_format),
        ("Examples", examples),
        ("Query", query_part)
    ]
    for title, content in checklist_items:
        if content and len(content.strip()) > 0:
            prompt += "- [ ] **{}**\n{}\n\n".format(title, content.strip())
    return prompt.strip()

def checklist_format_extractor(prompt: str) -> str:
    cleaned_text = re.sub(r'- \[ \] \*\*[^*]+\*\*\n', '', prompt).strip()

    return cleaned_text

def narrative_flow_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    components = [task_instruction, task_detail, output_format, examples, query_part]
    narrative_flow = ""
    for component in components:
        if component and len(component) > 0:
            narrative_flow += component.strip() + " "
    return narrative_flow.strip()

def narrative_flow_extractor(prompt: str) -> str:
    return prompt.strip()

def guided_steps_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    prompt = ""
    steps = [("1. Understand the Task", task_instruction),
             ("2. Detailing the Task", task_detail),
             ("3. Learning Through Examples", examples),
             ("4. Apply Your Knowledge", query_part)]
    for title, content in steps:
        if content and len(content) > 0:
            prompt += f"{title}\n{content.strip()}\n\n"
    return prompt.strip()

def guided_steps_extractor(prompt: str) -> str:
    cleaned_text = re.sub(r'\d+\. [^\n]+\n', '', prompt)
    return cleaned_text.strip()

def visual_infographic_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    components = [
        ("Task Instruction Icon", task_instruction),
        ("Task Detail Icon", task_detail),
        ("Output Format Icon", output_format),
        ("Examples Icon", examples + "\n\n" + query_part)
    ]
    infographic = ""
    for icon, text in components:
        if text and len(text) > 0:
            infographic += f"{icon}\n**{text.strip()}**\n\n"
    return infographic.strip()

def visual_infographic_extractor(prompt: str) -> str:
    
    cleaned_text = re.sub(r'[^\n]+ Icon\n\*\*', '', prompt)
    cleaned_text = re.sub(r'\*\*\n', '', cleaned_text)
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    return cleaned_text.strip()

def progressive_disclosure_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    prompt_elements = [task_instruction, task_detail, output_format, examples, query_part]
    prompt = ""
    for element in prompt_elements:
        if element and len(element) > 0:
            prompt += element.strip() + "\n\n"
    return prompt.strip()

def progressive_disclosure_extractor(component_text: str) -> str:
    return component_text.strip()


def dual_column_summary_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    prompt = ""
    left_column = ""
    right_column = ""
    for item, label in [(task_instruction, "Task Instruction"), (task_detail, "Task Detail")]:
        if item and len(item) > 0:
            left_column += f"{label}: {item.strip()}\n\n"
    for item, label in [(output_format, "Output Format"), (examples, "Examples"), (query_part, "Query")]:
        if item and len(item) > 0:
            right_column += f"{label}: {item.strip()}\n\n"
    if left_column and right_column:
        prompt = f"Left Column (Instruction & Detail):\n{left_column.strip()}\n\nRight Column (Output & Examples):\n{right_column.strip()}"
    return prompt


def dual_column_summary_extractor(prompt: str) -> str:
    left_column_content = re.search(r'Left Column \(Instruction & Detail\):\n(.*?)(?=\n\nRight Column \(Output & Examples\):|\Z)', prompt, re.DOTALL)
    right_column_content = re.search(r'Right Column \(Output & Examples\):\n(.*?)(?=\n\nQuery:|\Z)', prompt, re.DOTALL)

    def remove_labels(text: str) -> str:
        return re.sub(r'^[^:\n]*:\s*', '', text, flags=re.MULTILINE).strip()
    
    extracted = ""

    if left_column_content:
        left_text = left_column_content.group(1)
        extracted += remove_labels(left_text) + "\n\n"
        
    if right_column_content:
        right_text = right_column_content.group(1)
        extracted += remove_labels(right_text)
    
    return extracted.strip()

def concise_tabular_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    rendered_content = ""
    sections = [("Task Instruction", task_instruction), ("Task Detail", task_detail), ("Output Format", output_format), ("Examples", examples + "\n\n" + query_part)]    
    for title, content in sections:
        if content and len(content) > 0:
            rendered_content += f"| {title} | {content.strip()} |\n"    
    return rendered_content.strip()

def concise_tabular_extractor(prompt: str) -> str:    
    lines = re.findall(r'\|\s*[^|]+\s*\|\s*(.*?)\s*\|', prompt, re.DOTALL)
    extracted_content = "\n\n".join(lines).strip()
    
    return extracted_content


def hierarchical_bullet_points_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    elements = {
        "Task Instruction": task_instruction,
        "Task Detail": task_detail,
        "Output Format": output_format,
        "Examples": examples,
        "Query": query_part
    }    
    result = ""
    for key, value in elements.items():
        if value and len(value) > 0:
            result += f"- **{key}**\n"
            for item in value.split('\n'):
                if item.strip():
                    result += f"  - {item.strip()}\n"
    return result.strip()

def hierarchical_bullet_points_extractor(prompt: str) -> str:
    extracted_content = ""
    
    matches = re.finditer(r'- \*\*([^*]+)\*\*\n(?:  - (.*?))(?=\n- \*\*[^*]+\*\*|$)', prompt, re.DOTALL)
    
    for match in matches:
        title = match.group(1).strip()
        content = match.group(2).strip()
        if title and content:
            extracted_content += f"{content}\n"
    
    return extracted_content.strip()


def Interactive_FAQ_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    prompt = "##### FAQ: Understanding the Mathematical Problem\n\n"
    sections = [("What do I need to know before starting the mathematical problem?", task_instruction),
                ("Can you explain the task in more detail?", task_detail),
                ("How should I format my output?", output_format),
                ("Can you give me some examples?", examples),
                ("Query", query_part)]
    for title, content in sections:
        if content:
            prompt += f"**{title}**\n{content}\n\n"
    return prompt.strip()

def Interactive_FAQ_extractor(prompt: str) -> str:
    sections = re.findall(r'\*\*[^*]+\*\*\n(.*?)(?=\n\*\*[^*]+\*\*|$)', prompt, re.DOTALL)
    
    extracted_content = "\n\n".join(section.strip() for section in sections if section.strip())
    
    return extracted_content.strip()


def Simplified_Structured_Narrative_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    prompt = ""
    for item in [task_instruction, task_detail, output_format, examples, query_part]:
        if item:
            prompt += item.strip() + "\n\n"
    return prompt.strip()

def Simplified_Structured_Narrative_extractor(prompt: str):
    lines = [line.strip() for line in prompt.split('\n') if line.strip()]
    extracted_content = "\n\n".join(lines)
    
    return extracted_content.strip()


def narrative_guided_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    narrative = ""
    for item in [task_instruction, task_detail, output_format, examples, query_part]:
        if item:
            narrative += item.strip() + " "
    return narrative.strip()

def narrative_guided_extractor(narrative: str) -> str:
    return narrative.strip()

def narrative_flow_with_questions_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    prompt = ""
    components = [task_instruction, task_detail, output_format, examples, query_part]
    for component in components:
        if component and len(component) > 0:
            prompt += f"{component.strip()}\n\n"
    return prompt.strip()

def narrative_flow_with_questions_extractor(prompt: str) -> str:
    return prompt.strip()

def minimalistic_guide_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    sections = [("Task Instruction", task_instruction), ("Task Detail", task_detail), ("Output Format", output_format), ("Examples", examples), ("Query", query_part)]
    prompt = ""
    for title, content in sections:
        if content:
            prompt += f"{title}\n{content.strip()}\n\n"
    return prompt.strip()



def minimalistic_guide_extractor(component: str) -> str:
    return component.strip()

def guided_visual_outline_renderer(task_instruction: str, task_detail: str, output_format: str, examples: str, query_part: str) -> str:
    prompt = ""
    components = [("→ **Task Instruction**", task_instruction), ("→ **Task Detail**", task_detail), ("→ **Output Format**", output_format), ("→ **Examples**", examples + "\n\n" + query_part)]
    for title, content in components:
        if content and len(content) > 0:
            prompt += title + "\n" + content.strip().replace("\n", "\n► ") + "\n\n"
    return prompt.strip()

def guided_visual_outline_extractor(component_text: str) -> str:
    cleaned_content = re.sub(r'→ \*\*[^*]+\*\*\n', '', component_text)
    cleaned_content = cleaned_content.replace("► ", "")
    cleaned_content = re.sub(r'\n+', '\n', cleaned_content).strip()
    
    return cleaned_content


generated_prompt_renderer_pool = [
    (concise_bullet_points_renderer, concise_bullet_points_extractor),
    (structured_summary_renderer, structured_summary_extractor),
    (interactive_dialogue_renderer, interactive_dialogue_extractor),
    (simplified_markdown_bullet_points_renderer, simplified_markdown_bullet_points_extractor),
    (tabular_sections_renderer, tabular_sections_extractor),
    (descriptive_subheadings_renderer, descriptive_subheadings_extractor),
    (checklist_format_renderer, checklist_format_extractor),
    (narrative_flow_renderer, narrative_flow_extractor),
    (guided_steps_renderer, guided_steps_extractor),
    (visual_infographic_renderer, visual_infographic_extractor),
    (progressive_disclosure_renderer, progressive_disclosure_extractor),
    (dual_column_summary_renderer, dual_column_summary_extractor),
    (concise_tabular_renderer, concise_tabular_extractor),
    (hierarchical_bullet_points_renderer, hierarchical_bullet_points_extractor),
    (Interactive_FAQ_renderer, Interactive_FAQ_extractor),
    (Simplified_Structured_Narrative_renderer, Simplified_Structured_Narrative_extractor),
    (narrative_guided_renderer, narrative_guided_extractor),
    (narrative_flow_with_questions_renderer, narrative_flow_with_questions_extractor),
    (minimalistic_guide_renderer, minimalistic_guide_extractor),
    (guided_visual_outline_renderer, guided_visual_outline_extractor)
]

generated_prompt_format_desc_map = {
    concise_bullet_points_renderer: '',
    structured_summary_renderer: '',
    interactive_dialogue_renderer: '',
    simplified_markdown_bullet_points_renderer: '',
    tabular_sections_renderer: '',
    descriptive_subheadings_renderer: '',
    checklist_format_renderer: '',
    narrative_flow_renderer: '',
    guided_steps_renderer: '',
    visual_infographic_renderer: '',
    progressive_disclosure_renderer: '',
    dual_column_summary_renderer: '',
    concise_tabular_renderer: '',
    hierarchical_bullet_points_renderer: '',
    Interactive_FAQ_renderer: '',
    Simplified_Structured_Narrative_renderer: '',
    narrative_guided_renderer: '',
    narrative_flow_with_questions_renderer: '',
    minimalistic_guide_renderer: '',
    guided_visual_outline_renderer: '',
}


if __name__ == '__main__':
    task_instruction = "Write a function that returns the sum of two numbers."
    task_detail = "The function should take two numbers as input and return their sum."
    output_format = "int: the sum of the two numbers"
    examples = "Example 1:\nInput: 1, 2\nOutput: 3\n\nExample 2:\nInput: -1, 1\nOutput: 0"
    query_part = "Input: 1, 2\nOutput:"

    with open("prompt_format_examples.txt", "w") as file:
        for renderer_fn, extractor_fn in generated_prompt_format_pool:
            prompt = renderer_fn(task_instruction, task_detail, output_format, examples, query_part)
            file.write(f"Prompt format: {renderer_fn.__name__}\n")
            file.write("rendered prompt:\n")
            file.write(prompt + "\n\n")
            extracted = extractor_fn(prompt)
            file.write(f"Extracted:\n{extracted}\n\n")