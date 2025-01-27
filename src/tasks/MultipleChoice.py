import datasets
import re
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from .base import BaseTask
from liquid import Template
import logging
import random


class MultipleChoiceTask(BaseTask):
    def __init__(
        self,
        data_dir: str,
        train_size: int,
        minibatch_size: int,
        valid_size: int,
        test_size: int,
        answer_marker: Optional[str] = '',
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(data_dir, train_size, minibatch_size, valid_size, test_size, answer_marker, logger)
        self.task_intention = "solve a multiple choice task"
        self.num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        self.letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        self.dataset = self.load_task_dataset(data_dir)
        self.train_set, self.valid_set, self.test_set = self.dataset

    def load_task_dataset(self, data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load and split the dataset into train, validation, and test sets."""
        dataset = datasets.load_dataset(path="ai2_arc", name="ARC-Challenge")
        train_examples = self._pre_process(dataset["train"])
        test_examples = self._pre_process(dataset["test"])

        test_set = test_examples if self.test_size == -1 else test_examples[: self.test_size]

        train_size = len(train_examples)
        if self.valid_size > train_size:
            raise ValueError("valid_size cannot be greater than the number of train examples.")

        valid_indices = random.sample(range(train_size), self.valid_size)
        valid_set = [train_examples[i] for i in valid_indices]

        remaining_indices = [i for i in range(train_size) if i not in valid_indices]
        if self.train_size == -1:
            train_set = [train_examples[i] for i in remaining_indices]
        else:
            if self.train_size > len(remaining_indices):
                raise ValueError("train_size cannot be greater than the remaining train examples after validation.")
            train_set = [train_examples[i] for i in random.sample(remaining_indices, self.train_size)]

        return train_set, valid_set, test_set

    def _pre_process(self, dataset: datasets.Dataset) -> List[Dict]:
        """Preprocess the dataset by standardizing answer keys and shuffling."""
        out_doc = [
            {
                "id": doc["id"],
                "question": doc["question"],
                "choices": doc["choices"]["text"],
                "label": [self.num_to_letter.get(doc["answerKey"], doc["answerKey"])],
            }
            for doc in dataset
        ]
        random.shuffle(out_doc)
        return out_doc

    def extract_answer(self, response: str) -> str:
        """Extract the answer from the model's response."""
        response = response.strip().upper()
        if response[0] in self.letter_to_num:
            return response[0]
        if response in self.num_to_letter:
            return self.num_to_letter[response]
        return response

    def check_answer(self, pred: str, answer: List[str]) -> bool:
        """Check if the predicted answer matches the correct answer."""
        pred_answer = self.extract_answer(pred)
        return pred_answer in answer if pred_answer else False

    def run_evaluate(
        self, eval_llm, prompt, examples: List[Dict], desc: Optional[str] = None
    ) -> Tuple[float, List[Tuple[str, List[str]]], List[str], List[str], List[int]]:
        """Evaluate the model on the given examples."""
        texts, answers, score_list, prompts = [], [], [], []

        for example in examples:
            query_text = prompt.render_query(question=example["question"], choices=example["choices"])
            if not query_text:
                continue

            try:
                rendered_prompt = Template(str(prompt)).render(query=query_text)
                rendered_prompt = re.sub(r"\n{3,}", "\n\n", rendered_prompt)
                prompts.append(rendered_prompt)
            except Exception as e:
                self.logger.error(f"Error rendering prompt: {e}")
                continue

        preds = eval_llm.inference(prompts, use_batch_acceleration=True, desc=desc)

        for ex, pred in tqdm(zip(examples, preds), desc=desc, leave=True):
            texts.append((ex["question"], ex["choices"]))
            answers.append(ex["label"][0])
            score_list.append(int(self.check_answer(pred, ex["label"])))

        score = sum(score_list) / len(score_list) if score_list else 0
        return score, texts, answers, preds, score_list