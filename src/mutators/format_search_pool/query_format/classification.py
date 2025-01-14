import re

def TL_letter_renderer(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    return f"T: {question}\nL: {cot_hinter}{answer}"

def TL_letter_extractor(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'T: (.*?)\nL:', text, re.DOTALL)
    answers = re.findall(r'L: (.*)', text, re.DOTALL)

    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question,
            "answer": answer.strip(),
            }
        )
    return example_list

def TL_renderer(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip() # delete the space in query
    return f"Text: {question}\nLabel: {answer_with_hint}".strip()

def TL_extractor(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'Text: (.*?)\nLabel:', text, re.DOTALL)
    answers = re.findall(r'Label: (.*)', text, re.DOTALL)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def IO_renderer(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    return f"Input:\n{question}\nOutput:\n{cot_hinter}{answer}"

def IO_extractor(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'Input:\n(.*?)\nOutput:', text, re.DOTALL)
    answers = re.findall(r'Output:\n(.*)', text, re.DOTALL)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question,
            "answer": answer.strip(),
            }
        )
    return example_list

def Conv_renderer(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    return f"Human: {question}\nAI: {cot_hinter}{answer}"

def Conv_extractor(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'Human: (.*?)\nAI:', text, re.DOTALL)
    answers = re.findall(r'AI: (.*)', text, re.DOTALL)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question,
            "answer": answer.strip(),
            }
        )
    return example_list

def IR_renderer(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    return f"### Instruction:\n{question}\n\n### Response:\n{cot_hinter}{answer}"

def IR_extractor(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'### Instruction:\n(.*?)\n\n### Response:', text, re.DOTALL)
    answers = re.findall(r'### Response:\n(.*)', text, re.DOTALL)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question,
            "answer": answer.strip(),
            }
        )
    return example_list

classification_query_format_pool = [
    (TL_letter_renderer, TL_letter_extractor),
    (TL_renderer, TL_extractor),
    (IR_renderer, IR_extractor),
    (IO_renderer, IO_extractor),
    (Conv_renderer, Conv_extractor),
]