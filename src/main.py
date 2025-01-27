import os
import argparse
import importlib
from datetime import datetime
import logging
from optimizer import Optimizer
from prompt import PromptHistory
from mutators.case_diagnosis import CaseDiagnosis
from mutators.monte_carlo_sampling import MonteCarloSampling
from mutators.format_mutator import FormatMutator
from mutators.format_search_pool import SEARCH_POOL

# Configure logging
def configure_logging(output_path):
    # Create a logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler and a stream handler
    file_handler = logging.FileHandler(output_path)
    stream_handler = logging.StreamHandler()

    # Set the logging format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# Helper Functions
def get_output_marker(args):
    return f"{args.output_marker}-R_{args.rounds}-bm_size_{args.beam_size}-mb_size-v_sz_{args.valid_size}-t_sz_{args.test_size}"

def get_prompt_components(task_name):
    component_dict = {
        'case_diagnosis': ['TASK_INSTRUCTION', 'TASK_DETAIL', 'OUTPUT_FORMAT', 'EXAMPLES'],
        'monte_carlo_sampling': ['TASK_INSTRUCTION', 'TASK_DETAIL', 'OUTPUT_FORMAT', 'EXAMPLE_HINTER', 'EXAMPLES'],
        'all': ['TASK_INSTRUCTION', 'TASK_DETAIL', 'OUTPUT_FORMAT', 'EXAMPLE_HINTER', 'EXAMPLES', 'PROMPT_RENDERER', 'QUERY_FORMAT']
    }

    if task_name in ['GSM8K', 'MATH']:
        component_dict['monte_carlo_sampling'].append('COT_HINTER')
    return component_dict

def get_model_class(model_cls_name):
    if 'Llama' in model_cls_name or 'Mistral' in model_cls_name or 'Phi3' in model_cls_name:
        model_cls_name = 'Vllm'
    module_name = f"models.{model_cls_name}"
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, f'{model_cls_name}Model')
        return model_class
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Cannot find model named {model_cls_name} in module {module_name}") from e

def get_task_class(task_cls_name):
    module_name = f"tasks.{task_cls_name}"
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, f'{task_cls_name}Task')
        return model_class
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Cannot find model named {task_cls_name} in module {module_name}") from e


def get_prompt(task_cls_name):
    module_name = f"init_prompts.{task_cls_name}"
    try:
        module = importlib.import_module(module_name)
        prompt = module.prompt
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Cannot find prompt named {task_cls_name} in module {module_name}") from e
    return prompt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='GSM8K')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--output_marker', default="")
    parser.add_argument('--train_size', default=10, type=int)
    parser.add_argument('--minibatch_size', default=20, type=int)
    parser.add_argument('--valid_size', default=10, type=int)
    parser.add_argument('--test_size', default=10, type=int)
    parser.add_argument('--controller', default='multimute_4-linear_temp_0.7_beam-1')
    parser.add_argument('--opt_llm', default='GPT4')
    parser.add_argument('--eval_llm', default='Mistral')
    parser.add_argument('--vllm_pth', default='../Mistral-7B-v0.1')
    parser.add_argument('--init_temperature', default=1.0, type=float)
    parser.add_argument('--rounds', default=5, type=int)
    parser.add_argument('--beam_size', default=8, type=int)
    parser.add_argument('--num_return', default=4, type=int)
    parser.add_argument('--num_feedbacks', default=1, type=int)
    parser.add_argument('--errors_per_feedback', default=1, type=int)
    parser.add_argument('--correct_per_feedback', default=1, type=int)
    parser.add_argument('--apply_per_feedback', default=1, type=int, help='Number of improved results per feedback during applying')
    parser.add_argument('--num_random', default=1, type=int)
    parser.add_argument('--num_format', default=1, type=int)
    parser.add_argument('--select_method', default='UCT', type=str)
    parser.add_argument('--gpu_id', default='0', type=str)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Logging Configuration
    project_name = datetime.now().strftime("%b-%d-%H-%M-%S") + '-' + get_output_marker(args)
    output_folder = os.path.join('./result/', args.task, f'Opt_{args.opt_llm}-Eval_{args.eval_llm}', project_name)
    os.makedirs(output_folder, exist_ok=True)
    log_file_path = os.path.join(output_folder, 'output_log.txt')

    # Define the logger
    logger = configure_logging(log_file_path)

    logger.info(f"Log file: {log_file_path}")

    # Initialization
    prompt = get_prompt(args.task)
    logger.info(f"Initial Prompt: {prompt}")

    task = get_task_class(args.task)(data_dir=args.data_dir, train_size=args.train_size, valid_size=args.valid_size, test_size=args.test_size, minibatch_size=args.minibatch_size, answer_marker=" The answer is: ")
    component_dict = get_prompt_components(args.task)
    opt_llm = get_model_class(args.opt_llm)(max_tokens=4096)
    eval_llm = get_model_class(args.eval_llm)(model_path=args.vllm_pth, max_tokens=256, stop='\n', repetition_penalty=1.0)

    if args.task in ['MultipleChoice']:
        search_pool = SEARCH_POOL['MultiChoice']
    elif args.task in ['GSM8K', 'MATH']:
        search_pool = SEARCH_POOL['QA']
    elif args.task in ['BBH']:
        search_pool = SEARCH_POOL['Classification']

    prompt_history_path = '/mnt/teamdrive/yuanye/PromptHistory/'
    prompt_history = PromptHistory(init_prompt=prompt, root_path=prompt_history_path, init_round=0)

    # Mutators
    case_diagnosis = CaseDiagnosis(
        mutation_llm=opt_llm,
        eval_llm=eval_llm,
        task=task,
        num_error_per_feedback=args.errors_per_feedback,
        num_correct_per_feedback=args.correct_per_feedback,
        COMPONENT_KEYS=component_dict['case_diagnosis'],
        logger=logger,
    )

    monte_carlo_sampling = MonteCarloSampling(
        mutation_llm=opt_llm,
        task=task,
        COMPONENT_KEYS=component_dict['monte_carlo_sampling'],
        logger=logger,
    )

    format_mutator = FormatMutator(
        mutation_llm=opt_llm,
        task=task,
        COMPONENT_KEYS=['PROMPT_RENDERER', 'QUERY_FORMAT'],
        prompt_history=prompt_history,
        search_pool=search_pool,
        select_method = args.select_method,
        logger=logger,
    )

    num_prompts_per_round = {"case_diagnosis": args.num_feedbacks, "monte_carlo_sampling": args.num_random, "format": args.num_format}

    optimizer = Optimizer(
        task=task,
        mutator_list=[case_diagnosis, monte_carlo_sampling, format_mutator],
        cur_round=0,
        total_round=args.rounds,
        num_prompt_return=args.num_return,
        num_prompts_per_round=num_prompts_per_round,
        output_path=output_folder,
        opt_controller=args.controller,
        beam_size=args.beam_size,
        eval_llm=eval_llm,
        COMPONENT_KEYS=component_dict['all'],
        init_temperature=args.init_temperature,
        prompt_history=prompt_history,
        logger=logger,
        project_name = project_name
    )

    result = optimizer.run(init_prompt=prompt)