from utils import convert_seconds, stringify_dict
import wandb
import time
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

class Optimizer:
    def __init__(
        self,
        task,
        mutator_list: List,
        cur_round: int,
        total_round: int,
        num_prompt_return: int,
        num_prompts_per_round: Dict[str, int],
        output_path: Optional[str] = None,
        opt_controller: Optional[str] = None,
        beam_size: Optional[int] = None,
        eval_llm = None,
        COMPONENT_KEYS = None,
        init_temperature: float = 1,
        prompt_history=None,
        logger=None,
        project_name=None,
    ):
        self.opt_controller = self._init_controller(opt_controller)
        self.case_diagnosis, self.monte_carlo_sampling, self.format_mutator = mutator_list

        self.cur_round = cur_round
        self.total_round = total_round
        self.task = task
        self.num_prompt_return = num_prompt_return
        self.num_prompts_per_round = num_prompts_per_round
        self.output_path = output_path
        self.beam_size = beam_size
        self.eval_llm = eval_llm
        self.init_temperature = init_temperature
        self.prompt_history = prompt_history
        self.logger = logger
        self.COMPONENT_KEYS = COMPONENT_KEYS
        self.project_name = project_name

    def _init_controller(self, opt_controller: Optional[str]) -> "Controller":
        class Controller:
            def __init__(self, opt_controller: str):
                parts = opt_controller.split('-')
                self.mutate_scheduler = parts[0].strip()
                self.temp_scheduler = parts[1].strip()
                self.beam_scheduler = parts[2].strip()

        return Controller(opt_controller) if opt_controller else None

    def run(self, init_prompt) -> List:
        prompts = [init_prompt] if not isinstance(init_prompt, list) else init_prompt
        start_time = time.time()

        for round in range(self.cur_round, self.total_round + 1):
            self.round = round
            round_start_time = time.time()
            self._log_round_start(round, prompts)

            # ### Test format mutator
            # self.prompt_history.beam_history[-1] = prompts
            # prompts = self.format_mutator(prompts, self.num_prompts_per_round['format'], self.round)
            # break
            # ### Test format mutator
            
            if round == 0:
                self._evaluate_initial_round(prompts)
            else:
                prompts = self._process_round(prompts)

            self._evaluate_test_set(prompts, round)
            self._log_round_end(round, round_start_time)

        self._log_final_time(start_time)
        return prompts[:self.num_prompt_return]

    def _log_round_start(self, round: int, prompts: List):
        """Log the start of a round."""
        self.logger.info(f'\n\n\n##############################\n##############################\n####### ROUND {round} START! Prompts Length: {len(prompts)}#######\n##############################\n##############################\n')

    def _evaluate_initial_round(self, prompts: List):
        """Evaluate the initial round."""
        self.logger.info(f"\n================ In Round {self.round}. Start Evaluation on valid set ================")
        score, _, _, _, _ = self.task.run_evaluate(self.eval_llm, prompts[0], self.task.valid_set, desc='Run evaluate on valid set')
        self.prompt_history.beam_history[self.round] = [prompts[0]]

    def _process_round(self, prompts: List):
        """Process a round of optimization."""
        if self.num_prompts_per_round['format'] > 0 and self.format_mutator:
            if self.round % 2 == 1:
                prompts = self._expand_and_score_diagnosis_variation(prompts)
            else:
                prompts = self._expand_and_score_format(prompts)
            self._update_format_pool()
        else:
            prompts = self._expand_and_score_diagnosis_variation(prompts)

        self._update_prompt_history(prompts)
        return prompts

    def _expand_and_score_diagnosis_variation(self, prompts: List) -> List:
        """Expand and score prompts using feedback and random mutators."""
        start_time = time.time()
        self.logger.info(f"\n================ In Round {self.round}. Start Expand Candidates by Feedback Mutator and Random Mutators================")
        prompts, minibatch = self.expand_candidates_diagnosis_variation(prompts)
        self.logger.info(f'\n ROUND {self.round} FEEDBACK AND RANDOM EXPAND TIME: {convert_seconds((time.time() - start_time))}\n')

        self.logger.info(f"\n================ In Round {self.round}. Start Score {len(prompts)} Candidates and Beam Search ================")
        start_time = time.time()
        prompts, _ = self.score_candidates(prompts)
        self.logger.info(f'\n ROUND {self.round} SCORE TIME: {convert_seconds((time.time() - start_time))}\n')
        return prompts

    def _expand_and_score_format(self, prompts: List) -> List:
        """Expand and score prompts using format mutator."""
        self.logger.info(f"\n================ In Round {self.round}. Start Expand Candidates by Format Mutator================")
        start_time = time.time()
        prompts = self.expand_candidates_format(prompts)
        self.logger.info(f'\n ROUND {self.round} FORMAT EXPAND TIME: {convert_seconds((time.time() - start_time))}\n')

        self.logger.info(f"\n================ In Round {self.round}. Start Score {len(prompts)} Candidates and Beam Search ================")
        start_time = time.time()
        prompts, _ = self.score_candidates(prompts)
        self.logger.info(f'\n ROUND {self.round} SCORE TIME: {convert_seconds((time.time() - start_time))}\n')
        return prompts

    def _update_format_pool(self):
        """Update the format pool."""
        self.logger.info(f"\n================ In Round {self.round}. Start Update Format Pool ================")
        self.format_mutator.update_format_pool(self.round)
        self.prompt_history.format_pool[self.round] = deepcopy(self.format_mutator.format_pool)
        self.logger.info(f"\n================ Format pool ================\n{stringify_dict(self.format_mutator.format_pool)}")

    def _update_prompt_history(self, prompts: List):
        self.logger.info(f"\n================ In Round {self.round}. Start Update Prompt History ================")
        self.prompt_history.beam_history[self.round] = prompts
        self.prompt_history.round = self.round
        self.prompt_history.save(path=self.project_name)

    def _evaluate_test_set(self, prompts: List, round: int):
        """Evaluate prompts on the test set."""
        self.logger.info(f"\n================ In Round {self.round}. Start Evaluation on test set ================")
        start_time = time.time()

        for rank, prompt in enumerate(prompts):
            self._log_and_evaluate_prompt(prompt, rank, round)

        self.logger.info(f'\n ROUND {round} EVALUATION TIME: {convert_seconds((time.time() - start_time))}\n')

    def _log_and_evaluate_prompt(self, prompt, rank: int, round: int):
        """Log and evaluate a single prompt."""
        self.logger.info(f"\n================ Round {round} Rank {rank} Candidate ================\n\n{prompt}")
        if prompt.test_score is None:
            test_score, _, _, _, _ = self.task.run_evaluate(self.eval_llm, prompt, self.task.test_set, desc='Run evaluate on test set')
            prompt.test_score = test_score
        else:
            test_score = prompt.test_score

        self.logger.info(f'Evaluate Score: {prompt.eval_score}, Test Score: {test_score}')
        self._log_to_wandb(prompt, rank, round)

    def _log_to_wandb(self, prompt, rank: int, round: int):
        """Log prompt details to WandB."""
        wandb_log = {
            "round": round,
            "rank": rank,
            "prompt_round": prompt.round,
            "eval_score": prompt.eval_score,
            "test_score": prompt.test_score,
            "improved_score": prompt.improved_score,
            "action_desc": prompt.action_desc,
            "action_detail": prompt.action_detail,
            "whole_prompt": str(prompt),
        }

        for attribute in self.COMPONENT_KEYS:
            if 'FORMAT' in attribute and attribute != 'OUTPUT_FORMAT':
                wandb_log[attribute] = [getattr(prompt, attribute.lower())[0].__name__, getattr(prompt, attribute.lower())[1].__name__]
            elif attribute == 'EXAMPLES':
                for i, example in enumerate(prompt.examples):
                    wandb_log[f'example_{i}'] = prompt.render_one_example(example=example)
            else:
                wandb_log[attribute] = getattr(prompt, attribute.lower())

        run = wandb.init(project=self.project_name, name=f'Round_{round}-Rank{rank}', reinit=True, config=wandb_log, id=f'Round_{round}-Rank{rank}')
        run.finish()

    def _log_round_end(self, round: int, round_start_time: float):
        """Log the end of a round."""
        self.logger.info(f'\n ROUND {round} OVERALL TIME: {convert_seconds((time.time() - round_start_time))}\n')

    def _log_final_time(self, start_time: float):
        """Log the total time taken."""
        self.logger.info(f'\nFINISHED! OVERALL TIME: {convert_seconds((time.time() - start_time))}\n')

    def expand_candidates_diagnosis_variation(self, prompts: List) -> Tuple[List, List]:
        """Expand prompts using feedback and random mutators."""
        temperature = self.get_temperature()
        num_component = self.get_num_mutations()

        self.logger.info(f"\n--------- Curr Round: {self.round}, Curr prompts length: {len(prompts)} to expand, Curr temperature: {temperature}, Curr Mutate Component number: {num_component}\n")

        minibatch = self.task.sample_minibatch()
        new_prompts = []

        for i, prompt in enumerate(prompts):
            self.logger.info(f"\n-------- In Round {self.round}. Start to expand {i} prompt through feedback and random mutators --------\n")
            new_prompts_per_prompt = [prompt]
            if self.num_prompts_per_round['case_diagnosis'] > 0:
                new_prompts_per_prompt += self.case_diagnosis(prompt, minibatch, self.num_prompts_per_round['case_diagnosis'], num_component, self.round, temperature)
            if self.num_prompts_per_round['monte_carlo_sampling'] > 0:
                new_prompts_per_prompt += self.monte_carlo_sampling(prompt, self.num_prompts_per_round['monte_carlo_sampling'], num_component, self.round, temperature)
            new_prompts.append(new_prompts_per_prompt)

        return new_prompts, minibatch

    def expand_candidates_format(self, prompts: List) -> List:
        """Expand prompts using the format mutator."""
        if self.num_prompts_per_round['format'] > 0:
            self.logger.info(f"\n--------- Curr Round: {self.round}, Curr prompts length: {len(prompts)} to expand\n")
            return self.format_mutator(prompts, self.num_prompts_per_round['format'], self.round)
        return [[prompt] for prompt in prompts]

    def score_candidates(self, prompts: List) -> Tuple[List, List]:
        """Score a list of prompts."""
        prompts = [item for sublist in prompts for item in sublist]  # Flatten list
        scores = []

        for prompt in prompts:
            if prompt.eval_score is None:
                score, _, _, _, _ = self.task.run_evaluate(self.eval_llm, prompt, self.task.valid_set, desc='Run evaluate on valid set')
                prompt.eval_score = score
                prompt.improved_score = score - prompt.parent.eval_score if prompt.parent else None
            else:
                score = prompt.eval_score
            scores.append(score)

        combined = list(zip(prompts, scores))
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)[:self.beam_size]
        sorted_prompts, sorted_scores = zip(*sorted_combined)

        self.logger.info(f"Round {self.round} Number of selected prompts: {len(sorted_prompts)}")
        return list(sorted_prompts), list(sorted_scores)

    def get_temperature(self) -> float:
        """Get the current temperature based on the scheduler."""
        if self.opt_controller.temp_scheduler == 'linear_temp_0.7':
            return self.init_temperature - (self.round / self.total_round) * (self.init_temperature - 0.7)
        elif self.opt_controller.temp_scheduler == 'linear_temp_0.5':
            return self.init_temperature - (self.round / self.total_round) * (self.init_temperature - 0.5)
        elif self.opt_controller.temp_scheduler == 'exp_temp':
            return self.init_temperature * (0.1 / self.init_temperature) ** (self.round / self.total_round)
        elif self.opt_controller.temp_scheduler == 'linear_temp':
            return self.init_temperature - (self.round / self.total_round) * (self.init_temperature - 0.1)
        return self.init_temperature

    def get_num_mutations(self) -> int:
        """Get the number of mutations based on the scheduler."""
        steps_list = {
            "multimute_1": [4, 4, 3, 3, 2, 2, 1, 1, 1, 1],
            "multimute_2": [4, 4, 4, 3, 3, 2, 2, 1, 1, 1],
            "multimute_3": [4, 3, 2, 1, 1, 1, 1, 1, 1, 1],
            "multimute_4": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "multimute_5": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }.get(self.opt_controller.mutate_scheduler, [1] * (self.total_round + 1))

        if len(steps_list) - 1 < self.total_round:
            steps_list.extend([steps_list[-1]] * (self.total_round + 1 - len(steps_list)))

        return steps_list[self.round // 2]