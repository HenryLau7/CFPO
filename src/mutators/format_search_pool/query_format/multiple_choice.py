import re

def plain_renderer(question, choices, answer='', cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    option_str = ' '.join([f'{x}. {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"{question}\n{option_str}\nAnswer: {cot_hinter}{answer}".strip()

def plain_extractor(text, cot_hinter):
    example_list = []
    questions = re.findall(r'(.*?)\n[A-E]\.', text)
    options = re.findall(r'[A-D]\.\s*(.*?)(?=[A-D]\.|$)', text)
    answers = re.findall(r'Answer: (.*)', text)
    
    options = [options]
    for question, option, answer in zip(questions, options, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question.strip(),
            "choices": [item.strip() for item in option],
            "answer": answer.strip()
            }
        )
    return example_list

def QA_renderer(question, choices,  answer='', cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    option_str = '\n'.join([f'{x}: {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"Question: {question}\n{option_str}\nAnswer: {cot_hinter.strip()}{answer}".strip()
    
def QA_extractor(text, cot_hinter):
    example_list = []
    questions = re.findall(r'Question: (.*?)\n[A-E]:', text)
    options = re.findall(r'[A-D]: (.+)', text)
    options = [options]
    answers = re.findall(r'Answer: (.*)', text)
    
    for question, option, answer in zip(questions, options, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question.strip(),
            "choices": [item.strip() for item in option],
            "answer": answer
            }
        )
    return example_list


def QA_renderer_2(question, choices,  answer='', cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    option_str = 'Choices:\n'+'\n'.join([f'{x}: {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"Question: {question}\n{option_str}\nAnswer: {cot_hinter.strip()}{answer}".strip()
    
def QA_extractor_2(text, cot_hinter):
    example_list = []
    questions = re.findall(r'Question: (.*?)\nChoices', text)
    options = re.findall(r'Choices:\n(.*?)\nAnswer:', text, re.DOTALL)
    answers = re.findall(r'Answer: (.*)', text)
    
    for question, option, answer in zip(questions, options, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question.strip(),
            "choices": [item[3:] for item in option.split('\n')],
            "answer": answer
            }
        )
    return example_list

def markdown_renderer(question, choices, answer='', cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    option_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"### Question: {question}\n\n### Options:\n{option_str}\n\n### Answer: {cot_hinter}{answer}".strip()


def markdown_extractor(text, cot_hinter):
    example_list = []
    questions = re.findall(r'### Question: (.*?)\n\n### Options:', text)
    options = re.findall(r'### Options:\n(.*?)\n\n### Answer:', text, re.DOTALL)
    answers = re.findall(r'### Answer: (.*)', text)
    
    for question, option, answer in zip(questions, options, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question,
            "choices": [item[4:] for item in option.split('\n')],
            "answer": answer
            }
        )
    return example_list

multiple_choice_query_format_pool = [
    (plain_renderer, plain_extractor),
    (markdown_renderer, markdown_extractor),
    (QA_renderer, QA_extractor),
    (QA_renderer_2, QA_extractor_2),
]

query_format_desc_map_MultiChoice = {
    QA_renderer: "This renderer organizes content into \"Question\", \"Options\", and \"Answer\" sections, each clearly marked with headers. The question is placed under \"Question\", and the answer under \"Answer\". The options are listed under \"Options\", each prefixed with a letter (e.g., A, B, C) for easy reference. A hint, if included, is integrated directly before the answer, maintaining a clear structural and thematic division.",
    markdown_renderer: "This renderer adopts a Markdown-style formatting, with the question, options, and answer presented in a structured and visually appealing manner. The question is placed under \"Question\", the options under \"Options\", and the answer under \"Answer\". Each option is prefixed with a letter (e.g., A, B, C) for easy reference. A hint, if included, is integrated directly before the answer, maintaining a clear structural and thematic division.",
}


if __name__ == "__main__":
    question = "You are pushing a truck along a road. Would it be easier to accelerate this truck on Mars? Why? (Assume there is no friction)"
    choice = ['Choice 1', 'Choice 2', 'Choice 3', 'Choice 4']
    answer = 'D'
