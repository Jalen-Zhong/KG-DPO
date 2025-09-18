# paths of data and models
DS_HOTPOT=./data/eval/Hotpot_test_mc.json
DS_PQA=./data/eval/PQA_test_mc.json

function hotpot_eval(){
    python ./eval/evaluate_mcq_multi.py $DS_HOTPOT \
        --base_url https://YOUR_API_URL \
        --api_key YOUR_API_KEY \
        --model hotpot-post-training-qwen2.5-1.5b-base \
        --max_workers 32 \
        --option_num 10 \
        --num_runs 10 \
        --exp_name hotpot-dpo-1p5b-base-full-corpus-qa-none-rejected-1vs20
}

function pqa_eval(){
    python ./eval/evaluate_mcq_multi.py $DS_PQA \
        --base_url https://YOUR_API_URL \
        --api_key YOUR_API_KEY \
        --model hotpot-post-training-qwen2.5-1.5b-base \
        --max_workers 16 \
        --option_num 6 \
        --num_runs 10 \
        --exp_name pqa-dpo-1p5b-instruct-full-corpus-qa-none-rejected-1vs20
}


hotpot_eval \
&& pqa_eval \
&& echo "eval done"

