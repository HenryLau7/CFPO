from .base import BaseTask
import datasets
import random
import re
from tqdm import tqdm
from liquid import Template
from typing import List, Dict, Tuple, Optional
import logging


class GSM8KTask(BaseTask):
    def __init__(
        self,
        data_dir: str,
        train_size: int,
        minibatch_size: int,
        valid_size: int,
        test_size: int,
        answer_marker: str = "The answer is",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the GSM8K task.
        """
        super().__init__(data_dir, train_size, minibatch_size, valid_size, test_size, answer_marker, logger)
        self.task_intention = "solve a reasoning task and answer the following mathematical problem"
        self.dataset = self.load_task_dataset(data_dir)
        self.train_set, self.valid_set, self.test_set = self.dataset


    def load_task_dataset(self, data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load and preprocess the GSM8K dataset.
        """
        dataset = datasets.load_dataset(path='openai/gsm8k', name='main')
        train_examples = self._pre_process(dataset["train"])
        test_examples = self._pre_process(dataset['test'])

        # Split dataset into train, validation, and test sets
        test_set = test_examples if self.test_size == -1 else test_examples[:self.test_size]

        train_size = len(train_examples)
        if self.valid_size > train_size:
            raise ValueError("valid_size is greater than the number of train examples.")

        valid_indices = random.sample(range(train_size), self.valid_size)
        valid_set = [train_examples[i] for i in valid_indices]

        remaining_train_indices = [i for i in range(train_size) if i not in valid_indices]
        if self.train_size == -1:
            train_set = [train_examples[i] for i in remaining_train_indices]
        else:
            if self.train_size > len(remaining_train_indices):
                raise ValueError("train_size is greater than the remaining number of train examples after validation set selection.")
            train_set = [train_examples[i] for i in random.sample(remaining_train_indices, self.train_size)]

        return train_set, valid_set, test_set

    def _pre_process(self, dataset) -> List[Dict]:
        """
        Preprocess the dataset.
        """
        out_doc = []
        for doc in dataset:
            label = doc['answer'].split('####')[-1].strip()
            text = doc['answer'].split('####')[0].strip()

            lines = text.split('\n')
            processed_lines = [f"{line.strip()}." if not line.strip().endswith('.') else line.strip() for line in lines]
            processed_text = ' '.join(processed_lines).strip()

            answer = f"{processed_text} {self.answer_marker} {label}."
            question = re.sub(r'\s+', ' ', doc['question'])
            answer = re.sub(r'\s+', ' ', answer)

            out_doc.append({"question": question, "answer": answer})
        return out_doc

    def _is_number(self, s: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a string is a number and return its numeric value.
        """
        try:
            return True, str(float(s))
        except ValueError:
            try:
                import unicodedata
                return True, str(unicodedata.numeric(s))
            except (TypeError, ValueError):
                return False, None

    def extract_answer(self, completion: str) -> Optional[str]:
        """
        Extract the answer from the model completion.
        """
        if not completion:
            return None

        preds = completion.split(self.answer_marker)
        pred = preds[1] if len(preds) > 1 else preds[-1]

        pred = pred.replace(",", "")
        numbers = re.findall(r"-?\d+\.?\d*", pred)

        if not numbers:
            return None

        pred = numbers[-1].rstrip(".")
        is_number, pred = self._is_number(pred)

        return pred if is_number else None

    def check_answer(self, pred: str, answer: str, tolerance: float = 1e-6) -> bool:
        """
        Check if the predicted answer matches the ground truth.
        """
        pred_answer = self.extract_answer(pred)
        gt_label = self.extract_answer(answer)

        if pred_answer is None or gt_label is None:
            return False

        try:
            return abs(float(pred_answer) - float(gt_label)) <= tolerance
        except ValueError:
            return False

    def run_evaluate(
        self,
        eval_llm,
        prompt,
        examples: List[Dict],
        desc: Optional[str] = None,
    ) -> Tuple[float, List[str], List[str], List[str], List[int]]:
        """
        Evaluate the model on the given examples.
        """
        questions, answers, score_list, prompts = [], [], [], []

        for example in examples:
            query_text = prompt.render_query(question=example['question'])
            if not query_text:
                continue

            try:
                temp_prompt = Template(str(prompt)).render(query=query_text)
                temp_prompt = re.sub(r'\n{3,}', '\n\n', temp_prompt)
                prompts.append(temp_prompt)
            except Exception as e:
                self.logger.error(f"Error rendering prompt: {e}")
                continue

        preds = eval_llm.inference(prompts, use_batch_acceleration=True, desc=desc)

        for ex, pred in tqdm(zip(examples, preds), desc=desc, leave=True):
            questions.append(ex['question'])
            answers.append(ex['answer'])
            score = int(self.check_answer(pred, ex['answer']))
            score_list.append(score)

        score = sum(score_list) / len(score_list) if score_list else 0
        return score, questions, answers, preds, score_list