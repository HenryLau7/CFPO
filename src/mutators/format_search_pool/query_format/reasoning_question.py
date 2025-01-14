import re

def QA_letter_renderer(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    return f"Q: {question}\nA: {cot_hinter}{answer}".strip()

def QA_letter_extractor(text, cot_hinter=''):
    example_list = []
    
    pattern = r'Q: (.*?)\nA: (.*?)(?=Q:|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    for question, answer in matches:
        if cot_hinter and cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question.strip(),
            "answer": answer.strip(),
            }
        )
    return example_list


def QA_renderer(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip() 
    return f"Question: {question}\nAnswer: {answer_with_hint}" 

def QA_extractor(text, cot_hinter=''):
    example_list = []
    pattern = r'Question: (.*?)\nAnswer: (.*?)(?=Question:|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    for question, answer in matches:
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
    pattern = r'Input:\n(.*?)\nOutput:\n(.*?)(?=\nInput:|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for question, answer in matches:
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question.strip(),
            "answer": answer.strip(),
            }
        )
    return example_list

def Conv_renderer(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    return f"Human: {question}\nAI: {cot_hinter}{answer}"

def Conv_extractor(text, cot_hinter=''):
    example_list = []
    pattern = r'Human: (.*?)\nAI: (.*?)(?=\nHuman:|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for question, answer in matches:
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question.strip(),
            "answer": answer.strip(),
            }
        )
    return example_list

def IR_renderer(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    return f"### Instruction:\n{question}\n\n### Response:\n{cot_hinter}{answer}"

def IR_extractor(text, cot_hinter=''):
    example_list = []
    pattern = r'### Instruction:\n(.*?)\n### Response:\n(.*?)(?=\n### Instruction:|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    for question, answer in matches:
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
            "question": question.strip(),
            "answer": answer.strip(),
            }
        )
    return example_list

QA_query_format_pool = [
    (QA_letter_renderer, QA_letter_extractor),
    (QA_renderer, QA_extractor),
    (IR_renderer, IR_extractor),
    (IO_renderer, IO_extractor),
    (Conv_renderer, Conv_extractor),
]

query_format_desc_map_QA = {
    QA_letter_renderer: "This renderer adopts a minimalistic approach by directly presenting the question followed by the answer, prefixed with \"A:\". If a hint is involved, it is placed directly before the answer. The straightforward and unembellished format focuses solely on content delivery without additional structuring cues.",
    QA_renderer: "This renderer organizes content into \"Question\" and \"Answer\" sections, each clearly marked with headers. The question is placed under \"Question\", and the answer under \"Answer\". A hint, if included, is integrated directly before the answer, maintaining a clear structural and thematic division.",
    IO_renderer: "This renderer divides content into two distinct sections: \"Input\" and \"Output\". It clearly labels the question under \"Input\" and the answer under \"Output\". If a hint is provided, it is seamlessly integrated into the output section, maintaining a clear distinction between the question and the answer. This format is ideal for showcasing how inputs are transformed into outputs.",
    Conv_renderer: "Designed to mimic a conversational format, this renderer labels the question as \"Human:\" and the answer as \"AI:\". The inclusion of a hint (if any) is neatly integrated before the answer, separated by a space. This style is particularly useful for emulating a dialogue or interaction scenario.",
    IR_renderer: "This renderer organizes content into \"Instruction\" and \"Response\" sections, each clearly marked with headers. The question is placed under \"Instruction\", and the answer under \"Response\". A hint, if included, is integrated directly before the answer, maintaining a clear structural and thematic division.",
}