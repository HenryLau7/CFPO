# Script to run GSM8K

current_time=$(date +"%Y%m%d%H%M%S")

python src/main.py \
    --task GSM8K \
    --output_marker GSM8K_MISTRAL \
    --train_size -1 \
    --minibatch_size 40 \
    --valid_size 500 \
    --test_size -1 \
    --controller multimute_1-linear_temp_0.7-beam_1 \
    --opt_llm GPT4 \
    --eval_llm Mistral \
    --vllm_pth ../Mistral-7B-v0.1 \
    --init_temperature 1.0 \
    --rounds 40 \
    --beam_size 8 \
    --num_return 2 \
    --num_feedbacks 4 \
    --errors_per_feedback 5 \
    --correct_per_feedback 5 \
    --apply_per_feedback 1 \
    --num_random 4 \
    --num_format 8 \
    --num_knowledge 0\
    --gpu_id 0