import re

def query_renderer_QA_TITLECASE_SEPARATOR(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip()
    return f"Question || {question}\nAnswer || {answer_with_hint}"

def query_extractor_QA_TITLECASE_SEPARATOR(text, cot_hinter=''):
    import re
    example_list = []
    questions = re.findall(r'Question \|\| (.*?)\nAnswer \|\|', text)
    answers = re.findall(r'Answer \|\| (.*)', text)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CAPS_ARROW(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip() # Ensure no leading or trailing spaces
    return f"QUESTION: {question}\n=> ANSWER: {answer_with_hint}"

def query_extractor_QA_CAPS_ARROW(text, cot_hinter=''):
    import re
    example_list = []
    questions_answers = re.findall(r'QUESTION: (.*?)\n=> ANSWER: (.*)', text, re.DOTALL)
    for question, answer in questions_answers:
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_LOWERCASE_hyphen(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip() # ensure any cot_hinter appears correctly formatted with the answer
    return f"question: {question} - answer: {answer_with_hint}"

def query_extractor_QA_LOWERCASE_hyphen(text, cot_hinter=''):
    import re
    example_list = []
    matches = re.findall(r'question: (.*?) - answer: (.*)', text)
    for question, answer in matches:
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip() # clean the cot_hinter if present
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_MIXEDCASE_slash(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip()
    return f"Question: {question} / ANSWER: {answer_with_hint}"

def query_extractor_QA_MIXEDCASE_slash(text, cot_hinter=''):
    import re
    example_list = []
    occurrences = re.findall(r'Question: (.*?) / ANSWER: (.*)', text)
    for question, answer in occurrences:
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CAPS_COLON_SPACE(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip()
    return f"QUESTION: {question}\nANSWER: {answer_with_hint}"

def query_extractor_QA_CAPS_COLON_SPACE(text, cot_hinter=''):
    import re
    example_list = []
    questions = re.findall(r'QUESTION: (.*?)\nANSWER:', text)
    answers = re.findall(r'ANSWER: (.*)', text)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CAPS_UNDERSCORE(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip() # Ensure no extra space at the beginning
    return f"QUESTION: {question}\nANSWER: {answer_with_hint}"

def query_extractor_QA_CAPS_UNDERSCORE(text, cot_hinter=''):
    import re
    example_list = []
    questions = re.findall(r'QUESTION: (.*?)\nANSWER:', text)
    answers = re.findall(r'ANSWER: (.*)', text)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_TITLE_DASH_SPACE(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip() # delete the space in query
    return f"Question - {question}\nAnswer - {answer_with_hint}"

def query_extractor_QA_TITLE_DASH_SPACE(text, cot_hinter=''):
    import re
    example_list = []
    questions_answers = re.findall(r'Question - (.*?)\nAnswer - (.*)', text)
    for question, answer in questions_answers:
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CAPS_COLON_NEWLINE(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + "\n" if cot_hinter else ""
    return f"QUESTION: {question}\nANSWER:\n{cot_hinter}{answer}".strip()

def query_extractor_QA_CAPS_COLON_NEWLINE(text, cot_hinter=''):
    example_list = []
    questions_answers = re.split(r'QUESTION: ', text)[1:]  # Splitting by 'QUESTION: ' but skip the first split as it will be empty
    for qa in questions_answers:
        split_qa = qa.split("\nANSWER:\n")
        if len(split_qa) != 2:  # Check if both QUESTION and ANSWER are present
            continue
        question, answer = split_qa
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_BRACKETS_COLON_NEWLINE(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + "\n" if cot_hinter else ""
    return f"[Question]:\n{question}\n\n[Answer]:\n{cot_hinter}{answer}"

def query_extractor_QA_BRACKETS_COLON_NEWLINE(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'\[Question\]:\n(.*?)\n\n\[Answer\]:', text, re.DOTALL)
    answers = re.findall(r'\[Answer\]:\n(.*)', text, re.DOTALL)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_SmallCaps_ColonSpace(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    return f"question: {question}\nanswer: {cot_hinter}{answer}"

def query_extractor_QA_SmallCaps_ColonSpace(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'question: (.*?)\nanswer:', text, re.IGNORECASE)
    answers = re.findall(r'answer: (.*)', text, re.IGNORECASE)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CAPS_DOT_SPACE(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip() # delete the space in query
    return f"QUESTION. {question}\nANSWER. {answer_with_hint}"

def query_extractor_QA_CAPS_DOT_SPACE(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'QUESTION. (.*?)\nANSWER.', text)
    answers = re.findall(r'ANSWER. (.*)', text)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CapsBold_ColonNewline(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + "\n" if cot_hinter else ""
    return f"**QUESTION**:\n{question}\n\n**ANSWER**:\n{cot_hinter}{answer}"

def query_extractor_QA_CapsBold_ColonNewline(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'\*\*QUESTION\*\*:\n(.*?)\n\n\*\*ANSWER\*\*:', text, re.DOTALL)
    answers = re.findall(r'\*\*ANSWER\*\*:\n(.*)', text, re.DOTALL)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CapsSpace_EqualNewline(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    return f"QUESTION = {question}\nANSWER =\n{cot_hinter}{answer}"

def query_extractor_QA_CapsSpace_EqualNewline(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'QUESTION = (.*?)\nANSWER =', text)
    answers = re.findall(r'ANSWER =\n(.*)', text, re.DOTALL)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CapsItalic_ColonDoubleSpace(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + "  " if cot_hinter else ""
    question_formatted = f"QUESTION: {question}"
    answer_with_hint = f"ANSWER: {cot_hinter}{answer}".strip()
    return f"{question_formatted}\n{answer_with_hint}"

def query_extractor_QA_CapsItalic_ColonDoubleSpace(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'QUESTION: (.*?)\nANSWER:', text)
    answers = re.findall(r'ANSWER: (.*)', text)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_BulletPoint_ColonSpace(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    return f"• Question: {question}\n• Answer: {cot_hinter}{answer}"

def query_extractor_QA_BulletPoint_ColonSpace(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'• Question: (.*?)\n• Answer:', text)
    answers = re.findall(r'• Answer: (.*)', text)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CapsBracket_SemicolonSpace(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip()
    return f"[QUESTION]: {question}\n[ANSWER]: {answer_with_hint}"

def query_extractor_QA_CapsBracket_SemicolonSpace(text, cot_hinter=''):
    import re
    example_list = []
    questions = re.findall(r'\[QUESTION\]: (.*?)\n\[ANSWER\]:', text)
    answers = re.findall(r'\[ANSWER\]: (.*)', text)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_MixedCase_AsteriskColonSpace(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    rendered_content = f"Question*: {question}\nANSWER*: {cot_hinter}{answer}".strip()
    return rendered_content

def query_extractor_QA_MixedCase_AsteriskColonSpace(text, cot_hinter=''):
    import re
    example_list = []
    pattern = re.compile(r'Question\*: (.*?)\nANSWER\*: (.*)', re.DOTALL)
    matches = pattern.findall(text)
    for match in matches:
        question, answer = match
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CapsDash_Newline(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    return f"QUESTION -\n{question}\nANSWER -\n{cot_hinter}{answer}"

def query_extractor_QA_CapsDash_Newline(text, cot_hinter=''):
    example_list = []
    questions = re.findall(r'QUESTION -\n(.*?)\nANSWER -', text, re.DOTALL)
    answers = re.findall(r'ANSWER -\n(.*)', text, re.DOTALL)
    for question, answer in zip(questions, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CapsArrow_BulletPoint(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + " " if cot_hinter else ""
    answer_with_hint = f"{cot_hinter}{answer}".strip()
    return f"• QUESTION: {question}\n• ANSWER -> {answer_with_hint}"

def query_extractor_QA_CapsArrow_BulletPoint(text, cot_hinter=''):
    example_list = []
    pattern = r'• QUESTION: (.*?)\n• ANSWER -> (.*)'
    findings = re.findall(pattern, text)
    for question, answer in findings:
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    return example_list

def query_renderer_QA_CapsSpace_ColonDoubleNewline(question, answer, cot_hinter=''):
    cot_hinter = cot_hinter.strip() + "  " if cot_hinter else ""
    return f"QUESTION: {question}\n\nANSWER: \n{cot_hinter}{answer}"

def query_extractor_QA_CapsSpace_ColonDoubleNewline(text, cot_hinter=''):
    import re
    example_list = []
    
    # Adjusting the regex to account for the `cot_hinter` part
    cot_hinter_regex = re.escape(cot_hinter.strip() + "  ") if cot_hinter else ''
    regex_pattern = rf'QUESTION: (.*?)\n\nANSWER: \n({cot_hinter_regex}.*?)(?=\nQUESTION: |$)'
    
    # Use regex to find question-answer pairs
    ques_ans_pairs = re.findall(regex_pattern, text, re.DOTALL)
    
    # Process the extracted question-answer pairs
    for question, answer in ques_ans_pairs:
        if cot_hinter and cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append({
            "question": question.strip(),
            "answer": answer.strip(),
        })
    
    return example_list


generated_query_format_pool = [
    (query_renderer_QA_TITLECASE_SEPARATOR, query_extractor_QA_TITLECASE_SEPARATOR),
    (query_renderer_QA_CAPS_ARROW, query_extractor_QA_CAPS_ARROW),
    (query_renderer_QA_LOWERCASE_hyphen, query_extractor_QA_LOWERCASE_hyphen),
    (query_renderer_QA_MIXEDCASE_slash, query_extractor_QA_MIXEDCASE_slash),
    (query_renderer_QA_CAPS_COLON_SPACE, query_extractor_QA_CAPS_COLON_SPACE),
    (query_renderer_QA_CAPS_UNDERSCORE, query_extractor_QA_CAPS_UNDERSCORE),
    (query_renderer_QA_TITLE_DASH_SPACE, query_extractor_QA_TITLE_DASH_SPACE),
    (query_renderer_QA_CAPS_COLON_NEWLINE, query_extractor_QA_CAPS_COLON_NEWLINE),
    (query_renderer_QA_BRACKETS_COLON_NEWLINE, query_extractor_QA_BRACKETS_COLON_NEWLINE),
    (query_renderer_QA_SmallCaps_ColonSpace, query_extractor_QA_SmallCaps_ColonSpace),
    (query_renderer_QA_CAPS_DOT_SPACE, query_extractor_QA_CAPS_DOT_SPACE),
    (query_renderer_QA_CapsBold_ColonNewline, query_extractor_QA_CapsBold_ColonNewline),
    (query_renderer_QA_CapsSpace_EqualNewline, query_extractor_QA_CapsSpace_EqualNewline),
    (query_renderer_QA_CapsItalic_ColonDoubleSpace, query_extractor_QA_CapsItalic_ColonDoubleSpace),
    (query_renderer_QA_BulletPoint_ColonSpace, query_extractor_QA_BulletPoint_ColonSpace),
    (query_renderer_QA_CapsBracket_SemicolonSpace, query_extractor_QA_CapsBracket_SemicolonSpace),
    (query_renderer_QA_MixedCase_AsteriskColonSpace, query_extractor_QA_MixedCase_AsteriskColonSpace),
    (query_renderer_QA_CapsDash_Newline, query_extractor_QA_CapsDash_Newline),
    (query_renderer_QA_CapsArrow_BulletPoint, query_extractor_QA_CapsArrow_BulletPoint),
    (query_renderer_QA_CapsSpace_ColonDoubleNewline, query_extractor_QA_CapsSpace_ColonDoubleNewline),
]

generated_query_format_pool_desc_map = {
    query_renderer_QA_TITLECASE_SEPARATOR:'',
    query_renderer_QA_CAPS_ARROW:'',
    query_renderer_QA_LOWERCASE_hyphen:'',
    query_renderer_QA_MIXEDCASE_slash:'',
    query_renderer_QA_CAPS_COLON_SPACE:'',
    query_renderer_QA_CAPS_UNDERSCORE:'',
    query_renderer_QA_TITLE_DASH_SPACE:'',
    query_renderer_QA_CAPS_COLON_NEWLINE:'',
    query_renderer_QA_BRACKETS_COLON_NEWLINE:'',
    query_renderer_QA_SmallCaps_ColonSpace:'',
    query_renderer_QA_CAPS_DOT_SPACE:'',
    query_renderer_QA_CapsBold_ColonNewline:'',
    query_renderer_QA_CapsSpace_EqualNewline:'',
    query_renderer_QA_CapsItalic_ColonDoubleSpace:'',
    query_renderer_QA_BulletPoint_ColonSpace:'',
    query_renderer_QA_CapsBracket_SemicolonSpace:'',
    query_renderer_QA_MixedCase_AsteriskColonSpace:'',
    query_renderer_QA_CapsDash_Newline:'',
    query_renderer_QA_CapsArrow_BulletPoint:'',
    query_renderer_QA_CapsSpace_ColonDoubleNewline:'',
}


if __name__ == '__main__':
    

    # prompt = checklist_format_renderer(task_instruction, task_detail, output_format, examples, query_part)
    # # print(prompt)
    # extracted_query_part = checklist_format_extractor(prompt)
    # print(extracted_query_part)

    with open("query_format_examples.txt", "w") as file:
        for renderer_fn, extractor_fn in generated_query_format_pool:
            question = "What is the capital of France?"
            answer = "Paris"
            cot_hinter = "Let's think step by step."
            prompt = renderer_fn(question, answer, cot_hinter)
            file.write(f"Prompt format: {renderer_fn.__name__}\n")
            file.write("rendered prompt:\n")
            file.write(prompt + "\n\n")
            extracted = extractor_fn(prompt, cot_hinter)
            file.write(f"Extracted:\n{extracted}\n\n")
            

    #         # # print(f"Prompt format: {renderer_fn.__name__}")
    #         # print("rendered prompt:")
    #         # print(prompt)
    #         # extracted = extractor_fn(prompt)
    #         # print(f"Extracted:\n {extracted}")