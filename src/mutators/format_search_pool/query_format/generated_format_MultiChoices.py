import re

def query_renderer_Bullet_Points_Capital_Separator(question, choices, answer='', cot_hinter=''):
    option_str = '\n- '.join(choices)
    return f"- QUESTION: {question}\n- OPTIONS: \n- {option_str}\n- ANSWER: {cot_hinter.strip()}{answer}".strip()

def query_extractor_Bullet_Points_Capital_Separator(text, cot_hinter):
    example_list = []
    
    question_pattern = re.compile(r'- QUESTION: (.*?)\n- OPTIONS:', re.DOTALL)
    options_pattern = re.compile(r'- OPTIONS: \n(.*?)\n- ANSWER:', re.DOTALL)
    answer_pattern = re.compile(r'- ANSWER: (.*)', re.DOTALL)
    
    question_match = question_pattern.search(text)
    options_match = options_pattern.search(text)
    answer_match = answer_pattern.search(text)
    
    if question_match and options_match and answer_match:
        question = question_match.group(1).strip()
        options_block = options_match.group(1).strip()
        answer = answer_match.group(1).strip()
        
        options = options_block.split('\n- ')
        if options[0].startswith('- '):
            options[0] = options[0][2:]  

        options = [option.strip() for option in options]
        
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        
        example_list.append({
            "question": question,
            "choices": options,
            "answer": answer
        })
    
    return example_list

def query_renderer_Simple_Underline_Separator(question, choices, answer='', cot_hinter=''):
    option_str = ' '.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"QUESTION_{question}\nOPTIONS_{option_str}\nANSWER_{cot_hinter.strip()}{answer}".strip()

def query_extractor_Simple_Underline_Separator(text, cot_hinter):
    
    example_list = []
    sections = re.split(r'QUESTION_|OPTIONS_|ANSWER_', text)
    sections = list(filter(None, sections))

    for i in range(0, len(sections), 3):
        question = sections[i].strip()
        options_str = sections[i + 1].strip()
        answer = sections[i + 2].strip()

        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()

        options = re.findall(r'\((.)\)\s(.*?)\s*(?=\([A-E]\)|$)', options_str)
        choices = [opt[1] for opt in options]

        example_list.append(
            {
                "question": question,
                "choices": choices,
                "answer": answer
            }
        )
    return example_list

def query_renderer_Block_Case_Double_Dash_Separator(question, choices, answer='', cot_hinter=''):
    options_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D'], choices)])
    return f"QUESTION -- {question}\nOPTIONS --\n{options_str}\nANSWER -- {cot_hinter.strip()}{answer}".strip()

def query_extractor_Block_Case_Double_Dash_Separator(text, cot_hinter):
    example_list = []
    pattern = r'QUESTION -- (.*?)\nOPTIONS --\n(.*?)\nANSWER -- (.*)'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        question, options_block, answer = match
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        options = options_block.strip().split('\n')
        choices = [option[4:] for option in options]
        example_list.append({
            "question": question.strip(),
            "choices": choices,
            "answer": answer
        })

    return example_list


def query_renderer_Cascading_Statements(question, choices, answer='', cot_hinter=''):
    option_str = '\n  '.join([f'-{x} {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"Question: {question}\n Options:\n  {option_str}\n   Answer: {cot_hinter.strip()}{answer}".strip()

def query_extractor_Cascading_Statements(text, cot_hinter):
    example_list = []
    questions = re.findall(r'Question: (.*?)\n Options:', text)
    options_raw = re.findall(r'Options:\n  (.*?)\n   Answer:', text, re.DOTALL)
    options = [re.findall(r'-[A-E] (.*?)$', option, re.MULTILINE) for option in options_raw]
    answers = re.findall(r'Answer: (.*)', text)

    for question, option, answer in zip(questions, options, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
                "question": question,
                "choices": option,
                "answer": answer
            }
        )
    return example_list

def query_renderer_Highlight_Separator_Case(question, choices, answer='', cot_hinter=''):
    option_str = ' '.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D'], choices)])
    return f"QUESTION > {question}\nOPTIONS > {option_str}\nANSWER > {cot_hinter.strip()}{answer}"



def query_extractor_Highlight_Separator_Case(text, cot_hinter):
    example_list = []
    pattern = r'QUESTION > (.*?)\nOPTIONS > (.*?)\nANSWER > (.*)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        question, options_str, answer = match
        options = re.findall(r'\((.)\) (.*?)(?=\s*\(\w\)|$)', options_str)
        
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        
        example_list.append({
            "question": question.strip(),
            "choices": [option for _, option in options],
            "answer": answer.strip()
        })
    
    return example_list


def query_renderer_Simple_List_Separator(question, choices, answer='', cot_hinter=''):
    options_str = '\n- '.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D'], choices)])
    return f"Question: {question}; \nOptions: \n- {options_str}; \nAnswer: {cot_hinter.strip()}{answer}".strip()



def query_extractor_Simple_List_Separator(text, cot_hinter):
    example_list = []
    pattern = r'Question: (.*?); \nOptions: \n- (.*?); \nAnswer: (.*)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        question, options_block, answer = match
        options = re.findall(r'\((.)\) (.*?)(?=\s*\n- \(|;|$)', options_block)
        cleaned_answer = answer.replace(cot_hinter, '').strip()
        
        example_list.append({
            "question": question.strip(),
            "choices": [option for _, option in options],
            "answer": cleaned_answer
        })
    
    return example_list

def query_renderer_Indented_List_Separator(question, choices, answer='', cot_hinter=''):
    option_str = '\n\t* '.join([''] + [f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D'], choices)])
    return f"Question -> {question}\nOptions ->{option_str}\nAnswer -> {cot_hinter.strip()}{answer}".strip()




def query_extractor_Indented_List_Separator(text, cot_hinter):
    example_list = []
    pattern = r'Question -> (.*?)\nOptions ->\n\t\* (.*?)\nAnswer -> (.*)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        question, options_str, answer = match
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()

        options = re.findall(r'\(\w\) (.*?)(?=\n\t\* \(|\nAnswer ->|$)', options_str, re.DOTALL)
        
        example_list.append({
            "question": question.strip(),
            "choices": options,
            "answer": answer.strip()
        })
    
    return example_list


def query_renderer_Dash_Header_Simple_Content(question, choices, answer='', cot_hinter=''):
    option_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D'], choices)])
    return f"-- Question: {question} --\n-- Options: --\n{option_str}\n-- Answer: --\n{cot_hinter.strip()}{answer}".strip()

def query_extractor_Dash_Header_Simple_Content(text, cot_hinter):
    example_list = []
    questions = re.findall(r'-- Question: (.*?) --', text)
    options_blocks = re.findall(r'-- Options: --\n((?:\(.\) .*\n?)*)', text)
    answers = re.findall(r'-- Answer: --\n(.*)', text)

    for question, option_block, answer in zip(questions, options_blocks, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        options = [line[4:] for line in option_block.strip().split('\n')]
        example_list.append(
            {
                "question": question,
                "choices": options,
                "answer": answer
            }
        )
    return example_list

def query_renderer_TripleColon_Separator_CapitalHeader(question, choices, answer='', cot_hinter=''):
    option_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"Question::: {question}\nOptions:::\n{option_str}\nAnswer::: {cot_hinter.strip()}{answer}".strip()

def query_extractor_TripleColon_Separator_CapitalHeader(text, cot_hinter):
    example_list = []
    questions = re.findall(r'Question::: (.*?)\nOptions:::', text)
    options = re.findall(r'Options:::\n(.*?)\nAnswer:::', text, re.DOTALL)
    answers = re.findall(r'Answer::: (.*)', text)

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

def query_renderer_TwoDots_Separation_TitleCasing(question, choices, answer='', cot_hinter=''):
    option_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"Question:: {question}\nOptions::\n{option_str}\nAnswer:: {cot_hinter.strip()}{answer}".strip()




def query_extractor_TwoDots_Separation_TitleCasing(text, cot_hinter):
    example_list = []
    
    question_pattern = re.compile(r'Question:: (.*?)\nOptions::', re.DOTALL)
    options_pattern = re.compile(r'Options::\n(.*?)\nAnswer::', re.DOTALL)
    answer_pattern = re.compile(r'Answer:: (.*)', re.DOTALL)
    
    question_match = question_pattern.search(text)
    options_match = options_pattern.search(text)
    answer_match = answer_pattern.search(text)
    
    if question_match and options_match and answer_match:
        question = question_match.group(1).strip()
        options_str = options_match.group(1).strip()
        answer = answer_match.group(1).strip()
        
        options = options_str.split('\n')
        
        choices = [option[4:].strip() for option in options if option.startswith('(')]
        
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        
        example_list.append({
            "question": question,
            "choices": choices,
            "answer": answer
        })
    else:
        raise ValueError("Text format does not match expected pattern")
    
    return example_list

def query_renderer_NumericList_LabelCasing(question, choices, answer='', cot_hinter=''):
    options_str = '\n'.join([f'{x}) {y}' for x, y in zip(['A', 'B', 'C', 'D'], choices)])
    return f"1. Question: {question}\n\n2. Options:\n{options_str}\n\n3. Answer: {cot_hinter.strip()}{answer}".strip()


def query_extractor_NumericList_LabelCasing(text, cot_hinter):
    
    example_list = []
    questions = re.findall(r'1\. Question: (.*?)\n\n2\. Options:', text)
    options = re.findall(r'2\. Options:\n(.*?)\n\n3\. Answer:', text, re.DOTALL)
    answers = re.findall(r'3\. Answer: (.*)', text)
  
    for question, option_str, answer in zip(questions, options, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()      
        options_list = [item[3:].strip() for item in option_str.split('\n')]
        example_list.append(
            {
                "question": question.strip(),
                "choices": options_list,
                "answer": answer.strip()
            }
        )
    return example_list


def query_renderer_Highlight_Box_Separator(question, choices, answer='', cot_hinter=''):
    option_str = ' | \n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D'], choices)])
    return f"[QUESTION] | {question}\n[OPTIONS] | \n{option_str}\n[ANSWER] | {cot_hinter.strip()}{answer}".strip()

def query_extractor_Highlight_Box_Separator(text, cot_hinter):
    
    example_list = []
    questions = re.findall(r'\[QUESTION\] \| (.*?)\n\[OPTIONS\]', text)
    options = re.findall(r'\[OPTIONS\] \| \n(.*?)\n\[ANSWER\]', text, re.DOTALL)
    answers = re.findall(r'\[ANSWER\] \| (.*)', text)
    
    for question, option, answer in zip(questions, options, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
            formatted_options = [item[item.index(')')+2:] for item in option.split(' | \n')]
            example_list.append(
                {
                    "question": question,
                    "choices": formatted_options,
                    "answer": answer
                }
            )
    return example_list

def query_renderer_Bold_Separator_SentenceCasing(question, choices, answer='', cot_hinter=''):
    option_str = '\n'.join([f'- ({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"**Question** : {question}\n**Options** :\n{option_str}\n**Answer** : {cot_hinter.strip()}{answer}".strip()



def query_extractor_Bold_Separator_SentenceCasing(text, cot_hinter):
    example_list = []
    
    question_pattern = r'\*\*Question\*\* : (.*?)\n\*\*Options\*\* :'
    questions = re.findall(question_pattern, text, re.DOTALL)
    
    options_pattern = r'\*\*Options\*\* :\n(.*?)\n\*\*Answer\*\* :'
    options_blocks = re.findall(options_pattern, text, re.DOTALL)
    
    answers_pattern = r'\*\*Answer\*\* : (.*)'
    answers = re.findall(answers_pattern, text)
    
    for question, options_block, answer in zip(questions, options_blocks, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        
        options = re.findall(r'- \((.)\) (.*)', options_block)
        options = [option[1].strip() for option in options]
        
        example_list.append({
            "question": question.strip(),
            "choices": options,
            "answer": answer.strip()
        })
    
    return example_list


def query_renderer_Divided_CamelCase(question, choices, answer='', cot_hinter=''):
    option_str = ' || '.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D'], choices)])
    return f"Question || {question} ||\nOptions || {option_str} ||\nAnswer || {cot_hinter.strip()}{answer}".strip()


def query_extractor_Divided_CamelCase(text, cot_hinter):
    example_list = []
    
    question_pattern = r'Question \|\| (.*?) \|\|'
    options_pattern = r'Options \|\| (.*?) \|\|\nAnswer'
    answer_pattern = r'Answer \|\| (.*)$'
    
    question_match = re.search(question_pattern, text, re.DOTALL)
    options_match = re.search(options_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    if question_match and options_match and answer_match:
        question = question_match.group(1).strip()
        options_str = options_match.group(1).strip()
        answer_with_hint = answer_match.group(1).strip()
        
        options = re.findall(r'\((.)\) ([^|]+)', options_str)
        choices = [option[1].strip() for option in options]
        
        answer = re.search(rf'^\s*{cot_hinter}\s*(.*)', answer_with_hint)
        if answer:
            answer = answer.group(1).strip()
        else:
            answer = answer_with_hint.strip()
        
        example_list.append({
            "question": question,
            "choices": choices,
            "answer": answer
        })
    
    return example_list

def query_renderer_SimpleHash_Separator(question, choices, answer='', cot_hinter=''):
    option_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"Question# {question}\nOptions#\n{option_str}\nAnswer# {cot_hinter.strip()}{answer}".strip()

def query_extractor_SimpleHash_Separator(text, cot_hinter):
    
    example_list = []
    questions = re.findall(r'Question# (.*?)\nOptions#', text)
    options = re.findall(r'Options#\n(.*?)\nAnswer#', text, re.DOTALL)
    answers = re.findall(r'Answer# (.*)', text)

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

def query_renderer_DotBullet_CapitalHeader_SpaceSeparator(question, choices, answer='', cot_hinter=''):
    option_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"• QUESTION: {question}\n\n• OPTIONS:\n{option_str}\n\n• ANSWER: {cot_hinter.strip()}{answer}".strip()

def query_extractor_DotBullet_CapitalHeader_SpaceSeparator(text, cot_hinter):
    
    example_list = []
    questions = re.findall(r'• QUESTION: (.*?)\n\n• OPTIONS:', text)
    options = re.findall(r'• OPTIONS:\n(.*?)\n\n• ANSWER:', text, re.DOTALL)
    answers = re.findall(r'• ANSWER: (.*)', text)

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

def query_renderer_Highlighted_Title_Underdash_Separator(question, choices, answer='', cot_hinter=''):
    option_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D'], choices)])
    return f"**QUESTION_** {question}\n**OPTIONS_**\n{option_str}\n**ANSWER_** {cot_hinter.strip()}{answer}".strip()

def query_extractor_Highlighted_Title_Underdash_Separator(text, cot_hinter):
    example_list = []
    
    question_pattern = r'\*\*QUESTION_\*\* (.*?)\n\*\*OPTIONS_'
    options_pattern = r'\*\*OPTIONS_\*\*\n(.*?)\n\*\*ANSWER_'
    answer_pattern = r'\*\*ANSWER_\*\* (.*)'
    
    questions = re.findall(question_pattern, text, re.DOTALL)
    options_strs = re.findall(options_pattern, text, re.DOTALL)
    answers = re.findall(answer_pattern, text)
    
    for question, options_str, answer in zip(questions, options_strs, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        
        option_list = re.findall(r'\((.)\) (.+)', options_str)
        choices = [option[1].strip() for option in option_list]
        
        example_list.append({
            "question": question.strip(),
            "choices": choices,
            "answer": answer.strip()
        })
    
    return example_list


def query_renderer_DoubleBorder_CamelCaseSeparator(question, choices, answer='', cot_hinter=''):
    option_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"Question|| {question}\nOptions||\n{option_str}\nAnswer|| {cot_hinter.strip()}{answer}".strip()

def query_extractor_DoubleBorder_CamelCaseSeparator(text, cot_hinter):
    
    example_list = []
    questions = re.findall(r'Question\|\| (.*?)\nOptions\|\|', text)
    options = re.findall(r'Options\|\|\n(.*?)\nAnswer\|\|', text, re.DOTALL)
    answers = re.findall(r'Answer\|\| (.*)', text)

    for question, option, answer in zip(questions, options, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        example_list.append(
            {
                "question": question.strip(),
                "choices": [item[4:] for item in option.strip().split('\n')],
                "answer": answer
            }
        )
    return example_list

def query_renderer_Streamlined_DotDash(question, choices, answer='', cot_hinter=''):
    option_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f".Question: {question}\n.Options\n{option_str}\n---\n.Answer: {cot_hinter.strip()}{answer}".strip()

def query_extractor_Streamlined_DotDash(text, cot_hinter):
    example_list = []
    questions = re.findall(r'\.Question: (.*?)\n\.Options', text)
    options = re.findall(r'\.Options\n(.*?)\n---', text, re.DOTALL)
    answers = re.findall(r'\.Answer: (.*)', text)

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

def query_renderer_SimpleItalic_SemicolonSeparator(question, choices, answer='', cot_hinter=''):
    option_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D', 'E'], choices)])
    return f"*Question*: {question};\n\n*Options*:\n{option_str};\n\n*Answer*: {cot_hinter.strip()}{answer}".strip()

def query_extractor_SimpleItalic_SemicolonSeparator(text, cot_hinter):
    
    example_list = []
    questions = re.findall(r'\*Question\*: (.*?);', text)
    options = re.findall(r'\*Options\*:\n(.*?);', text, re.DOTALL)
    answers = re.findall(r'\*Answer\*: (.*)', text)

    for question, option, answer in zip(questions, options, answers):
        if cot_hinter in answer:
            answer = answer.replace(cot_hinter, '').strip()
        options_list = [item[4:].strip() for item in option.split('\n') if item.startswith('(')]
        example_list.append(
            {
                "question": question,
                "choices": options_list,
                "answer": answer
            }
        )
    return example_list

def query_renderer_Echoed_Title_ColonSpaceSeparator(question, choices, answer='', cot_hinter=''):
    option_str = '\n'.join([f'({x}) {y}' for x, y in zip(['A', 'B', 'C', 'D'], choices)])
    rendered_text = f"Question Echo: {question}\nOptions:\n{option_str}\nAnswer Echo: {cot_hinter.strip()}{answer}".strip()
    return rendered_text


def query_extractor_Echoed_Title_ColonSpaceSeparator(text, cot_hinter):
    
    example_list = []
    questions = re.findall(r'Question Echo: (.*?)\nOptions:', text)
    options = re.findall(r'Options:\n(.*?)\nAnswer Echo:', text, re.DOTALL)
    answers = re.findall(r'Answer Echo: (.*)', text)

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

def query_renderer_Inline_Separator_CapSense(question, choices, answer='', cot_hinter=''):
    options_str = ' '.join(['({}) {}'.format(x, y) for x, y in zip(['A', 'B', 'C', 'D'], choices)])
    rendered_query = "Question: {} | Options: {} | Answer: {}{}".format(question, options_str, cot_hinter.strip(), answer)
    return rendered_query.strip()



def query_extractor_Inline_Separator_CapSense(text, cot_hinter):
    example_list = []

    question_pattern = r'Question: (.*?)(?: \| Options:| \| Answer:)'
    options_pattern = r'Options: (.*?)(?: \| Answer:|$)'
    answer_pattern = r'Answer: (.*)$'

    question_match = re.search(question_pattern, text, re.DOTALL)
    if question_match:
        question = question_match.group(1).strip()
    else:
        raise ValueError("Question could not be extracted.")

    options_match = re.search(options_pattern, text, re.DOTALL)
    if options_match:
        options_str = options_match.group(1).strip()
        options = re.findall(r'\((.)\) (.+?)(?=\s*\(|\s*$)', options_str)
        choices = [opt[1].strip() for opt in options]
    else:
        raise ValueError("Options could not be extracted.")

    answer_match = re.search(answer_pattern, text, re.DOTALL)
    if answer_match:
        answer_str = answer_match.group(1).strip()
        if cot_hinter in answer_str:
            answer_str = answer_str.replace(cot_hinter, '').strip()
    else:
        raise ValueError("Answer could not be extracted.")

    example_list.append({
        "question": question,
        "choices": choices,
        "answer": answer_str
    })
    
    return example_list

generated_query_format_pool = [
    (query_renderer_Bullet_Points_Capital_Separator, query_extractor_Bullet_Points_Capital_Separator),
    (query_renderer_Simple_Underline_Separator, query_extractor_Simple_Underline_Separator),
    (query_renderer_Block_Case_Double_Dash_Separator, query_extractor_Block_Case_Double_Dash_Separator),
    (query_renderer_Cascading_Statements, query_extractor_Cascading_Statements),
    (query_renderer_Highlight_Separator_Case, query_extractor_Highlight_Separator_Case),
    (query_renderer_Simple_List_Separator, query_extractor_Simple_List_Separator),
    (query_renderer_Indented_List_Separator, query_extractor_Indented_List_Separator),
    (query_renderer_Dash_Header_Simple_Content, query_extractor_Dash_Header_Simple_Content),
    (query_renderer_TripleColon_Separator_CapitalHeader, query_extractor_TripleColon_Separator_CapitalHeader),
    (query_renderer_TwoDots_Separation_TitleCasing, query_extractor_TwoDots_Separation_TitleCasing),
    (query_renderer_NumericList_LabelCasing, query_extractor_NumericList_LabelCasing),
    (query_renderer_Highlight_Box_Separator, query_extractor_Highlight_Box_Separator),
    (query_renderer_Bold_Separator_SentenceCasing, query_extractor_Bold_Separator_SentenceCasing),
    (query_renderer_Divided_CamelCase, query_extractor_Divided_CamelCase),
    (query_renderer_SimpleHash_Separator, query_extractor_SimpleHash_Separator),
    (query_renderer_DotBullet_CapitalHeader_SpaceSeparator, query_extractor_DotBullet_CapitalHeader_SpaceSeparator),
    (query_renderer_Highlighted_Title_Underdash_Separator, query_extractor_Highlighted_Title_Underdash_Separator),
    (query_renderer_DoubleBorder_CamelCaseSeparator, query_extractor_DoubleBorder_CamelCaseSeparator),
    (query_renderer_Streamlined_DotDash, query_extractor_Streamlined_DotDash),
    (query_renderer_SimpleItalic_SemicolonSeparator, query_extractor_SimpleItalic_SemicolonSeparator),
    (query_renderer_Echoed_Title_ColonSpaceSeparator, query_extractor_Echoed_Title_ColonSpaceSeparator),
    (query_renderer_Inline_Separator_CapSense, query_extractor_Inline_Separator_CapSense),   
]

generated_query_format_pool_desc_map = {
    query_renderer_Bullet_Points_Capital_Separator: "Bullet_Points_Capital_Separator",
    query_renderer_Simple_Underline_Separator: "Simple_Underline_Separator",
    query_renderer_Block_Case_Double_Dash_Separator: "Block_Case_Double_Dash_Separator",
    query_renderer_Cascading_Statements: "Cascading_Statements",
    query_renderer_Highlight_Separator_Case: "Highlight_Separator_Case",
    query_renderer_Simple_List_Separator: "Simple_List_Separator",
    query_renderer_Indented_List_Separator: "Indented_List_Separator",
    query_renderer_Dash_Header_Simple_Content: "Dash_Header_Simple_Content",
    query_renderer_TripleColon_Separator_CapitalHeader: "TripleColon_Separator_CapitalHeader",
    query_renderer_TwoDots_Separation_TitleCasing: "TwoDots_Separation_TitleCasing",
    query_renderer_NumericList_LabelCasing: "NumericList_LabelCasing",
    query_renderer_Highlight_Box_Separator: "Highlight_Box_Separator",
    query_renderer_Bold_Separator_SentenceCasing: "Bold_Separator_SentenceCasing",
    query_renderer_Divided_CamelCase: "Divided_CamelCase",
    query_renderer_SimpleHash_Separator: "SimpleHash_Separator",
    query_renderer_DotBullet_CapitalHeader_SpaceSeparator: "DotBullet_CapitalHeader_SpaceSeparator",
    query_renderer_Highlighted_Title_Underdash_Separator: "Highlighted_Title_Underdash_Separator",
    query_renderer_DoubleBorder_CamelCaseSeparator: "DoubleBorder_CamelCaseSeparator",
    query_renderer_Streamlined_DotDash: "Streamlined_DotDash",
    query_renderer_SimpleItalic_SemicolonSeparator: "SimpleItalic_SemicolonSeparator",
    query_renderer_Echoed_Title_ColonSpaceSeparator: "Echoed_Title_ColonSpaceSeparator",
    query_renderer_Inline_Separator_CapSense: "Inline_Separator_CapSense"
}


if __name__ == '__main__':
    
    with open("query_format_examples.txt", "w") as file:
        for renderer_fn, extractor_fn in generated_query_format_pool:
            question = "Statement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements."
            choices = ["True, True",
                    "False, False",
                    "True, False",
                    "False, True"]
            answer = "C"
            cot_hinter = ""
            prompt = renderer_fn(question, choices,answer, cot_hinter)
            file.write(f"Prompt format: {renderer_fn.__name__}\n")
            file.write("rendered prompt:\n")
            file.write(prompt + "\n\n")
            extracted = extractor_fn(prompt, cot_hinter)
            file.write(f"Extracted:\n{extracted}\n\n")