import random
import logging
from typing import List, Dict, Tuple, Optional


class BaseTask:
    def __init__(
        self,
        data_dir: str,
        train_size: int,
        minibatch_size: int,
        valid_size: int,
        test_size: int,
        answer_marker: str = " The answer is ",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the base task.
        """
        self.minibatch_size = minibatch_size
        self.answer_marker = answer_marker
        self.task_intention = None
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.logger = logger if logger else logging.getLogger(__name__)

        self.dataset = None, None, None
        self.train_set, self.valid_set, self.test_set = self.dataset

    def isolate_answer(self, text: str, answer_marker: str) -> Optional[str]:
        """
        Isolate the answer from a given text using the answer marker.
        """
        if text is None:
            return None

        assert isinstance(text, str)
        text = text.lower()
        split_ans = text.split(self.answer_marker)
        if len(split_ans) > 1:
            ans = split_ans[-1].replace(":", "").strip()
            extract_ans_temp = ans.split(".\n")[0].strip()
            if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
                extract_ans = extract_ans_temp[:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip().strip("\n")
            return extract_ans
        else:
            return text

    def load_task_dataset(self, data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load and preprocess the dataset.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def sample_minibatch(self) -> List[Dict]:
        """
        Sample a minibatch from the training set.
        """
        minibatch = random.sample(self.train_set, k=min(self.minibatch_size, len(self.train_set)))
        return minibatch

    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract the answer from a response provided by a language model.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def check_answer(self, answer: str, ground_truth: str) -> bool:
        """
        Check if the response matches the ground truth.
        """
        raise NotImplementedError("Subclasses must implement this method.")