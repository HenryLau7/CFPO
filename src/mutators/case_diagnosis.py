from .base import BaseMutator
from tqdm import tqdm
import random
import re
from typing import List, Tuple, Optional
from utils import parse_tagged_text

class CaseDiagnosis(BaseMutator):
    def __init__(
        self,
        mutation_llm,
        task,
        num_error_per_feedback: int = 5,
        num_correct_per_feedback: int = 5,
        apply_per_feedback: int = 1,
        COMPONENT_KEYS: Optional[List[str]] = None,
        eval_llm=None,
        logger=None,
    ):
        super().__init__(mutation_llm, task, COMPONENT_KEYS)
        self.eval_llm = eval_llm
        self.apply_per_feedback = apply_per_feedback
        self.num_error_per_feedback = num_error_per_feedback
        self.num_correct_per_feedback = num_correct_per_feedback
        self.logger = logger
        self.component_desc = self.get_component_desc()

    def __call__(
        self, prompt, examples, num_prompts: int, num_component: int, round: int, temperature: float
    ) -> List:

        # Evaluate on the minibatch set
        score, questions, labels, preds, _ = self.task.run_evaluate(
            eval_llm=self.eval_llm, prompt=prompt, examples=examples, desc="Run evaluate on minibatch set"
        )

        # Get feedbacks from the mutation LLM
        feedbacks_list = self.get_feedbacks(prompt, num_prompts, num_component, questions, labels, preds, temperature)

        # Apply feedbacks to generate new prompts
        new_prompts = []
        for component_key_feedback in tqdm(feedbacks_list, desc="Applying feedbacks"):
            component_key_list, content_list, feedbacks = [], [], []
            for component_key, feedback in component_key_feedback:
                if component_key == "EXAMPLES":
                    content = self.apply_feedbacks_for_examples(prompt, feedback, self.apply_per_feedback, temperature)
                else:
                    content = self.apply_feedbacks(prompt, component_key, feedback, self.apply_per_feedback, temperature)

                component_key_list.append(component_key)
                content_list.append(content)
                feedbacks.append(feedback)

            new_prompt = prompt.generate(
                round=round,
                component_keys=component_key_list,
                component_contents=content_list,
                action_desc="case_diagnosis",
            )

            try:
                if str(new_prompt):
                    new_prompts.append(new_prompt)
            except Exception as e:
                self.logger.error(f"Failed to render new_prompt: {e}")
                continue

            # Log results
            self.logger.info(f"\n================ In Round {round} Get a new prompt via feedbacks ================")
            for i, (key, feedback) in enumerate(component_key_feedback):
                self.logger.info(f"## Component Key: {key}\n")
                self.logger.info(f"## Feedback {i}: {feedback}\n")
                self.logger.info(f"## New component: {str(content_list[i])}\n")
            self.logger.info(f"## Feedback summary: at most {num_component}, actual feedbacks {len(component_key_feedback)}\n")
            self.logger.info(f"## New prompt: {new_prompt}\n")

        return new_prompts

    def _parse_component(self, texts: List[str], prompt_component_pattern: str = r"<Prompt segment: (?:\d+\.)?\s*(.*?)>\n(.+)") -> List[Tuple[str, str]]:
        """Parse text that is tagged with start and end tags."""
        feedback = []
        for t in texts:
            try:
                component_key, reason = re.findall(prompt_component_pattern, t, re.DOTALL)[0]
                component_key = component_key.replace(" ", "_").upper()
                assert component_key in self.COMPONENT_KEYS
                feedback.append((component_key, reason))
            except Exception as e:
                if t:
                    self.logger.warning(f"Unexpected feedback format encountered: {t}\nError: {e}")
        return feedback

    def get_feedbacks(
        self, prompt, num_prompts: int, num_component: int, texts: List[str], labels: List[str], preds: List[str], temperature: float
    ) -> List[List[Tuple[str, str]]]:
        """Generate feedbacks for the prompt based on evaluation results."""
        feedbacks_list = []
        for _ in tqdm(range(num_prompts), desc="Get feedbacks..."):
            error_string = self._sample_error_str(texts, labels, preds, self.task, n=self.num_error_per_feedback)
            correct_string = self._sample_correct_str(texts, labels, preds, self.task, n=self.num_correct_per_feedback)
            if error_string is None and correct_string is None:
                continue
            feedbacks = self._get_feedbacks(prompt, error_string, correct_string, num_component, temperature)
            feedbacks_list.append(feedbacks)
        return feedbacks_list

    def _get_feedbacks(self, prompt, error_string: Optional[str], correct_string: Optional[str], num_component: int, temperature: float) -> List[Tuple[str, str]]:
        """Get feedbacks for a prompt based on evaluation results."""
        num_component_str = f"ONE" if num_component == 1 else f"AT MOST {num_component}"
        feedback_prompt = f"""{self._get_meta_prompt_header(prompt)}

        Upon evaluating the current prompt, this prompt gets the following examples wrong:
        {error_string or "None"}

        Meanwhile, this prompt gets the following examples correct:
        {correct_string or "None"}

        Please review the provided examples of correct and incorrect answers, and identify {num_component_str} specific area for improvement in the prompts. Each suggestion should focus on A SPECIFIC segment of the prompt that needs optimization. If you believe the EXAMPLES segment needs improvement, you may suggest one example that can be added, removed, or altered to enhance the EXAMPLES segment based on the examples given. If you think there is no need for improvement, do not return any prompt segment.
        Please encapsulate each suggestion using the following format:

        <START>
        <Prompt segment: [Segment name]>
        [Suggestion goes here]
        <END>
        """
        feedback_prompt = "\n".join([line.lstrip() for line in feedback_prompt.split("\n")])

        self.logger.info("\n================ Prompt to request feedbacks ================\n")
        self.logger.info(feedback_prompt)

        res = self.mutation_llm.inference(feedback_prompt, desc="Get feedbacks", temperature=temperature)
        feedback = self._parse_component(parse_tagged_text(res, "<START>", "<END>", logger=self.logger))

        if len(feedback) > num_component:
            feedback = feedback[:num_component]

        return feedback

    def apply_feedbacks(self, prompt, component_key: str, feedback_str: str, apply_per_feedback: int, temperature: float) -> Optional[str]:
        """Apply feedback to a specific component of the prompt."""
        prompt_to_apply_feedback = f"""{self._get_meta_prompt_header(prompt)}

        The existing {component_key} segment contains:
        \"\"\"{getattr(prompt, component_key.lower())}\"\"\"

        Here are some suggestions for improving the {component_key} segments: {feedback_str}

        Based on the above information, I wrote {apply_per_feedback} distinct and improved versions of the {component_key} segment within the prompt.
        Each revised segment is encapsulated between <START> and <END>. In case this segment is an empty string, generate a suitable one referring to the suggestion.
        The {apply_per_feedback} revised {component_key} segments are:
        """.strip()

        prompt_to_apply_feedback = "\n".join([line.lstrip() for line in prompt_to_apply_feedback.split("\n")])

        self.logger.info("\n================ Prompts to apply feedbacks ================\n")
        self.logger.info(prompt_to_apply_feedback)

        response = self.mutation_llm.inference(prompt_to_apply_feedback, desc="Apply feedbacks", temperature=temperature)
        new_prompt_components =parse_tagged_text(response, "<START>", "<END>", logger=self.logger)

        if new_prompt_components == [None]:
            return None

        return prompt.prompt_renderer[1](new_prompt_components[0])

    def apply_feedbacks_for_examples(self, prompt, feedback_str: str, apply_per_feedback: int, temperature: float) -> Optional[str]:
        """Apply feedback to the EXAMPLES segment of the prompt."""
        prompt_to_apply_feedback = f"""{self._get_meta_prompt_header(prompt)}

        The existing EXAMPLES segment contains:
        \"\"\"{prompt.render_examples(prompt.examples)}\"\"\"

        Here are some suggestions for enhancing the EXAMPLES segment: {feedback_str}

        Based on the above information, I have crafted {apply_per_feedback} improved version of the EXAMPLES segment within the prompt. Each revision represents ONLY ONE of the following specific actions:
        1. Addition: Incorporating one new example into the existing set.
        2. Deletion: Eliminating one single example from the current set.
        3. Modification: Changing the content of an example while maintaining its contextual relevance.
        Please present the results without indicating which action was taken. Each refined EXAMPLES segment is marked by <START> and <END>.

        The {apply_per_feedback} revised segments are:
        """
        prompt_to_apply_feedback = "\n".join([line.lstrip() for line in prompt_to_apply_feedback.split("\n")])

        self.logger.info("\n================ Prompts to apply feedbacks for examples ================\n")
        self.logger.info(prompt_to_apply_feedback)

        response = self.mutation_llm.inference(prompt_to_apply_feedback, desc="Apply feedbacks for examples", temperature=temperature)
        new_prompt_components = parse_tagged_text(response, "<START>", "<END>",logger = self.logger)

        if new_prompt_components == [None]:
            return None

        return prompt.query_format[1](new_prompt_components[0], prompt.cot_hinter)

    def _sample_error_str(self, texts: List[str], labels: List[str], preds: List[str], task, n: int = 4) -> Optional[str]:
        """Sample n error strings from the given texts, labels, and predictions."""
        error_idxs = [i for i, (pred, label) in enumerate(zip(preds, labels)) if not task.check_answer(pred, label)]
        if not error_idxs:
            return None

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))
        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]

        combined = list(zip(sample_texts, sample_labels, sample_preds))
        random.shuffle(combined)
        sample_texts, sample_labels, sample_preds = zip(*combined)

        error_string = ""
        for i, (t, a, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
            error_string += f"## Example {i + 1}\n"
            if task.__class__.__name__ in ["MATHTask"]:
                gt_label = task.extract_answer_from_gold_solution(a)
            else:
                gt_label = task.extract_answer(a)
            if task.__class__.__name__ in ["GSM8KTask", "MATHTask", "BBHTask"]:
                error_string += f"Question: {t.strip()}\nLabel: {gt_label}\nPrediction: {p.strip()}\n\n"
            elif task.__class__.__name__ in ["MMLUTask", "MultipleChoiceTask"]:
                question, choices = t
                error_string += f"Question: {question.strip()}\nChoices: {choices}\nLabel: {gt_label}\nPrediction: {p.strip()}\n\n"
            else:
                raise NotImplementedError

        return error_string.strip()

    def _sample_correct_str(self, texts: List[str], labels: List[str], preds: List[str], task, n: int = 4) -> Optional[str]:
        """Sample n correct strings from the given texts, labels, and predictions."""
        correct_idxs = [i for i, (pred, label) in enumerate(zip(preds, labels)) if task.check_answer(pred, label)]
        if not correct_idxs:
            return None

        sample_idxs = random.sample(correct_idxs, min(len(correct_idxs), n))
        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]

        combined = list(zip(sample_texts, sample_labels, sample_preds))
        random.shuffle(combined)
        sample_texts, sample_labels, sample_preds = zip(*combined)

        correct_string = ""
        for i, (t, a, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
            correct_string += f"## Example {i + 1}\n"
            if task.__class__.__name__ in ["MATHTask"]:
                gt_label = task.extract_answer_from_gold_solution(a)
            else:
                gt_label = task.extract_answer(a)
            if task.__class__.__name__ in ["GSM8KTask", "MATHTask", "BBHTask"]:
                correct_string += f"Question: {t.strip()}\nLabel: {gt_label}\nPrediction: {p.strip()}\n\n"
            elif task.__class__.__name__ in ["MMLUTask", "MultipleChoiceTask"]:
                question, choices = t
                correct_string += f"Question: {question.strip()}\nChoices: {choices}\nLabel: {gt_label}\nPrediction: {p.strip()}\n\n"
            else:
                raise NotImplementedError

        return correct_string.strip()