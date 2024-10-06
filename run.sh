CKPTS=(
    meta-llama/Meta-Llama-3-70B-Instruct
    meta-llama/Meta-Llama-3-8B-Instruct
    meta-llama/Llama-2-70b-chat-hf
    meta-llama/Llama-2-13b-chat-hf
    meta-llama/Llama-2-7b-chat-hf
    allenai/tulu-2-dpo-70b
    allenai/tulu-2-70b
    allenai/tulu-2-dpo-13b
    allenai/tulu-2-13b
    allenai/tulu-2-dpo-7b
    allenai/tulu-2-7b
    mistralai/Mistral-7B-Instruct-v0.1
    mistralai/Mistral-7B-Instruct-v0.2
    mistralai/Mixtral-8x7B-Instruct-v0.1
    01-ai/Yi-1.5-34B-Chat
    01-ai/Yi-1.5-9B-Chat
    Qwen/Qwen1.5-72B-Chat
    Qwen/Qwen1.5-32B-Chat
    Qwen/Qwen2-72B-Instruct
    CohereForAI/c4ai-command-r-plus
    CohereForAI/c4ai-command-r-v01
    google/gemma-7b-it
    google/gemma-2b-it
    mistralai/Mistral-7B-Instruct-v0.3
    microsoft/Phi-3-small-8k-instruct
    THUDM/glm-4-9b-chat
    google/gemma-2-9b-it
    google/gemma-2-27b-it
    meta-llama/Meta-Llama-3.1-70B-Instruct
    meta-llama/Meta-Llama-3.1-8B-Instruct
    Qwen/Qwen2.5-72B-Instruct
)  # model checkpoint

NAMES=(
    llama-3-70b
    llama-3-8b
    llama-2-70b
    llama-2-13b
    llama-2-7b
    tulu-2-dpo-70b
    tulu-2-70b
    tulu-2-dpo-13b
    tulu-2-13b
    tulu-2-dpo-7b
    tulu-2-7b
    mistral-7b-v0.1
    mistral-7b-v0.2
    mixtral-8x7b
    yi-1.5-34b
    yi-1.5-9b
    qwen-1.5-72b
    qwen-1.5-32b
    qwen-2-72b
    command-r-plus
    command-r-v01
    gemma-7b
    gemma-2b
    mistral-7b-v0.3
    phi-3-small
    glm-4-9b
    gemma-2-9b
    gemma-2-27b
    llama-3.1-70b
    llama-3.1-8b
    qwen-2.5-72b
)  # model name

CLSS=(
    llama3vllm
    llama3vllm
    llama2vllm
    llama2vllm
    llama2vllm
    tulu2vllm
    tulu2vllm
    tulu2vllm
    tulu2vllm
    tulu2vllm
    tulu2vllm
    mistralvllm
    mistralvllm
    mistralvllm
    hfvllm
    hfvllm
    hfvllm
    hfvllm
    hfvllm
    coherevllm
    coherevllm
    gemmavllm
    gemmavllm
    mistralvllm
    hfnosysvllm
    glmvllm
    gemmavllm
    gemmavllm
    llama3.1vllm
    llama3.1vllm
    hfvllm
)  # model class

CONFIG_DIRS=(
    configs/default/pairwise_base
    configs/default/pairwise_cot
    configs/default/pairwise_metric
    configs/default/pairwise_reference
    configs/default/pairwise_metric_reference
    configs/default/pairwise_synthesize
    configs/default/pairwise_cot_finegrained
    configs/default/pairwise_cot_finegrained_llama2
    configs/default/pairwise_chateval_round_1
    configs/default/pairwise_chateval_round_2
    configs/default/pairwise_chateval_round_2_llama2
    configs/default/pairwise_analysis_aggr_single_stage
    configs/default/pairwise_analysis_aggr_single_stage_llama2
    configs/default/pairwise_analysis_aggr
    configs/default/pairwise_analysis_aggr_llama2
    configs/default/pairwise_reference_gpt4
    configs/default/pairwise_prepair
    configs/default/sc/pairwise_cot_sc
    configs/default/sc/pairwise_protocol_consistency
)  # config dir

CONFIGS=(
    base
    cot
    metric
    reference
    metric-reference
    swap_and_synthesize
    fine_grained
    fine_grained_llama2
    multi-role-round1
    multi-role-round2
    multi-role-round2-llama2
    multi-aspect-single
    multi-aspect-single-llama2
    multi-aspect-two
    multi-aspect-two-llama2
    gpt4-reference
    prepair
    self-consistency
    protocol-consistency
)

ALL_SELECTED_MODELS=(
    llama-3-70b
    llama-3-8b
    llama-2-70b
    llama-2-13b
    llama-2-7b
    tulu-2-dpo-70b
    tulu-2-70b
    tulu-2-dpo-13b
    tulu-2-13b
    tulu-2-dpo-7b
    tulu-2-7b
    # mistral-7b-v0.1
    # mistral-7b-v0.2
    mixtral-8x7b
    yi-1.5-34b
    yi-1.5-9b
    qwen-1.5-72b
    qwen-1.5-32b
    qwen-2-72b
    # command-r-plus
    # command-r-v01
    gemma-7b
    gemma-2b
    mistral-7b-v0.3
    # phi-3-small
    glm-4-9b
    # gemma-2-9b
    # gemma-2-27b
    llama-3.1-70b
    llama-3.1-8b
    qwen-2.5-72b
)   # 24 base LLMs selected, llama-3.1-405b would require more than 8 GPUs


ALL_SELECTED_CONFIGS=(
    base
    cot
    metric
    reference
    metric-reference
    swap_and_synthesize
    fine_grained
    fine_grained_llama2
    multi-role-round1
    multi-role-round2
    multi-role-round2-llama2
    multi-aspect-single
    multi-aspect-single-llama2
    multi-aspect-two
    multi-aspect-two-llama2
    gpt4-reference
    prepair
    self-consistency
    protocol-consistency
)  # all configs for 15 evaluation protocols


GPU_IDS=0,1,2,3,4,5,6,7  # Change this to the GPU ids you want to use
NUM_GPUS=8  # Change this to the number of GPUs you want to use
PROTOCOL_FIRST=true

# Get results for LLMs that support >4K context length

SELECTED_MODELS=(
    llama-3-70b
    llama-3-8b
    tulu-2-dpo-70b
    tulu-2-70b
    tulu-2-dpo-13b
    tulu-2-13b
    tulu-2-dpo-7b
    tulu-2-7b
    mixtral-8x7b
    qwen-1.5-72b
    qwen-1.5-32b
    qwen-2-72b
    gemma-7b
    gemma-2b
    mistral-7b-v0.3
    glm-4-9b
    llama-3.1-70b
    llama-3.1-8b
    qwen-2.5-72b
)


SELECTED_CONFIGS=(
    base
    cot
    metric
    reference
    metric-reference
    swap_and_synthesize
    fine_grained
    multi-role-round1
    multi-role-round2
    multi-aspect-single
    multi-aspect-two
    gpt4-reference
    prepair
    self-consistency
)

echo number of GPUs $NUM_GPUS
echo number of ckpts ${#CKPTS[@]}
echo number of configs ${#CONFIGS[@]}
echo number of models ${#NAMES[@]}
echo number of clss ${#CLSS[@]}
echo number of selected models ${#SELECTED_MODELS[@]}
echo number of selected configs ${#SELECTED_CONFIGS[@]}
echo number of config dirs ${#CONFIG_DIRS[@]}


if [[ "${PROTOCOL_FIRST}" == "false" ]]; then
    # finishing the models first
    for model in "${SELECTED_MODELS[@]}"; do
        for i in "${!NAMES[@]}"; do
            if [[ "${NAMES[i]}" == "$model" ]]; then
                for config in "${SELECTED_CONFIGS[@]}"; do
                    for j in "${!CONFIGS[@]}"; do
                        if [[ "${CONFIGS[j]}" == "${config}" ]]; then
                            # run the forward pass
                            forward_config=${CONFIG_DIRS[$j]}.yaml
                            echo ${config} $forward_config ${NAMES[$i]} ${CKPTS[$i]} ${CLSS[$i]} 
                            RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=${GPU_IDS} python run.py \
                                --model_pt ${CKPTS[$i]} \
                                --model_name ${NAMES[$i]} \
                                --model_cls ${CLSS[$i]} \
                                --tensor_parallel_size $NUM_GPUS \
                                --gpu_memory_utilization 0.9 \
                                --batch_size 8 \
                                --swap_space 8 \
                                --download_dir $HOME/.cache/huggingface/hub \
                                --verbose \
                                --resume \
                                --config_dir $forward_config \
                                --use_cache \
                                --datasets llmbar_natural llmbar_adversarial mtbench instrusum
                            # run the swap pass
                            backward_config=${CONFIG_DIRS[$j]}_swap.yaml
                            echo ${config}_swap $backward_config ${NAMES[$i]} ${CKPTS[$i]} ${CLSS[$i]} 
                            RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=${GPU_IDS} python run.py \
                                --model_pt ${CKPTS[$i]} \
                                --model_name ${NAMES[$i]} \
                                --model_cls ${CLSS[$i]} \
                                --tensor_parallel_size $NUM_GPUS \
                                --gpu_memory_utilization 0.9 \
                                --batch_size 8 \
                                --swap_space 8 \
                                --download_dir $HOME/.cache/huggingface/hub \
                                --verbose \
                                --resume \
                                --config_dir $backward_config \
                                --use_cache \
                                --datasets llmbar_natural llmbar_adversarial mtbench instrusum
                            break
                        fi
                    done
                done
                break
            fi
        done
    done
else
    # finishing the configs first
    for config in "${SELECTED_CONFIGS[@]}"; do
        for j in "${!CONFIGS[@]}"; do
            if [[ "${CONFIGS[j]}" == "${config}" ]]; then
                for model in "${SELECTED_MODELS[@]}"; do
                    for i in "${!NAMES[@]}"; do
                        if [[ "${NAMES[i]}" == "$model" ]]; then
                            # run the forward pass
                            forward_config=${CONFIG_DIRS[$j]}.yaml
                            echo ${config} $forward_config ${NAMES[$i]} ${CKPTS[$i]} ${CLSS[$i]} 
                            RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=${GPU_IDS} python run.py \
                                --model_pt ${CKPTS[$i]} \
                                --model_name ${NAMES[$i]} \
                                --model_cls ${CLSS[$i]} \
                                --tensor_parallel_size $NUM_GPUS \
                                --gpu_memory_utilization 0.9 \
                                --batch_size 8 \
                                --swap_space 8 \
                                --download_dir $HOME/.cache/huggingface/hub \
                                --verbose \
                                --resume \
                                --config_dir $forward_config \
                                --use_cache \
                                --datasets llmbar_natural llmbar_adversarial mtbench instrusum
                            # run the swap pass
                            backward_config=${CONFIG_DIRS[$j]}_swap.yaml
                            echo ${config}_swap $backward_config ${NAMES[$i]} ${CKPTS[$i]} ${CLSS[$i]} 
                            RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=${GPU_IDS} python run.py \
                                --model_pt ${CKPTS[$i]} \
                                --model_name ${NAMES[$i]} \
                                --model_cls ${CLSS[$i]} \
                                --tensor_parallel_size $NUM_GPUS \
                                --gpu_memory_utilization 0.9 \
                                --batch_size 8 \
                                --swap_space 8 \
                                --download_dir $HOME/.cache/huggingface/hub \
                                --verbose \
                                --resume \
                                --config_dir $backward_config \
                                --use_cache \
                                --datasets llmbar_natural llmbar_adversarial mtbench instrusum
                            break
                        fi
                    done
                done
                break
            fi
        done
    done
fi


# Get results for LLMs that support 4K context length

SELECTED_MODELS=(
    llama-2-70b
    llama-2-13b
    llama-2-7b
)

SELECTED_CONFIGS=(
    base
    cot
    metric
    reference
    metric-reference
    swap_and_synthesize
    fine_grained_llama2
    multi-role-round1
    multi-role-round2-llama2
    multi-aspect-single-llama2
    multi-aspect-two-llama2
    gpt4-reference
    prepair
    self-consistency
    protocol-consistency
)

echo number of GPUs $NUM_GPUS
echo number of ckpts ${#CKPTS[@]}
echo number of configs ${#CONFIGS[@]}
echo number of models ${#NAMES[@]}
echo number of clss ${#CLSS[@]}
echo number of selected models ${#SELECTED_MODELS[@]}
echo number of selected configs ${#SELECTED_CONFIGS[@]}
echo number of config dirs ${#CONFIG_DIRS[@]}


if [[ "${PROTOCOL_FIRST}" == "false" ]]; then
    # finishing the models first
    for model in "${SELECTED_MODELS[@]}"; do
        for i in "${!NAMES[@]}"; do
            if [[ "${NAMES[i]}" == "$model" ]]; then
                for config in "${SELECTED_CONFIGS[@]}"; do
                    for j in "${!CONFIGS[@]}"; do
                        if [[ "${CONFIGS[j]}" == "${config}" ]]; then
                            # run the forward pass
                            forward_config=${CONFIG_DIRS[$j]}.yaml
                            echo ${config} $forward_config ${NAMES[$i]} ${CKPTS[$i]} ${CLSS[$i]} 
                            RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=${GPU_IDS} python run.py \
                                --model_pt ${CKPTS[$i]} \
                                --model_name ${NAMES[$i]} \
                                --model_cls ${CLSS[$i]} \
                                --tensor_parallel_size $NUM_GPUS \
                                --gpu_memory_utilization 0.9 \
                                --batch_size 8 \
                                --swap_space 8 \
                                --download_dir $HOME/.cache/huggingface/hub \
                                --verbose \
                                --resume \
                                --config_dir $forward_config \
                                --use_cache \
                                --datasets llmbar_natural llmbar_adversarial mtbench instrusum
                            # run the swap pass
                            backward_config=${CONFIG_DIRS[$j]}_swap.yaml
                            echo ${config}_swap $backward_config ${NAMES[$i]} ${CKPTS[$i]} ${CLSS[$i]} 
                            RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=${GPU_IDS} python run.py \
                                --model_pt ${CKPTS[$i]} \
                                --model_name ${NAMES[$i]} \
                                --model_cls ${CLSS[$i]} \
                                --tensor_parallel_size $NUM_GPUS \
                                --gpu_memory_utilization 0.9 \
                                --batch_size 8 \
                                --swap_space 8 \
                                --download_dir $HOME/.cache/huggingface/hub \
                                --verbose \
                                --resume \
                                --config_dir $backward_config \
                                --use_cache \
                                --datasets llmbar_natural llmbar_adversarial mtbench instrusum
                            break
                        fi
                    done
                done
                break
            fi
        done
    done
else
    # finishing the configs first
    for config in "${SELECTED_CONFIGS[@]}"; do
        for j in "${!CONFIGS[@]}"; do
            if [[ "${CONFIGS[j]}" == "${config}" ]]; then
                for model in "${SELECTED_MODELS[@]}"; do
                    for i in "${!NAMES[@]}"; do
                        if [[ "${NAMES[i]}" == "$model" ]]; then
                            # run the forward pass
                            forward_config=${CONFIG_DIRS[$j]}.yaml
                            echo ${config} $forward_config ${NAMES[$i]} ${CKPTS[$i]} ${CLSS[$i]} 
                            RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=${GPU_IDS} python run.py \
                                --model_pt ${CKPTS[$i]} \
                                --model_name ${NAMES[$i]} \
                                --model_cls ${CLSS[$i]} \
                                --tensor_parallel_size $NUM_GPUS \
                                --gpu_memory_utilization 0.9 \
                                --batch_size 8 \
                                --swap_space 8 \
                                --download_dir $HOME/.cache/huggingface/hub \
                                --verbose \
                                --resume \
                                --config_dir $forward_config \
                                --use_cache \
                                --datasets llmbar_natural llmbar_adversarial mtbench instrusum
                            # run the swap pass
                            backward_config=${CONFIG_DIRS[$j]}_swap.yaml
                            echo ${config}_swap $backward_config ${NAMES[$i]} ${CKPTS[$i]} ${CLSS[$i]} 
                            RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=${GPU_IDS} python run.py \
                                --model_pt ${CKPTS[$i]} \
                                --model_name ${NAMES[$i]} \
                                --model_cls ${CLSS[$i]} \
                                --tensor_parallel_size $NUM_GPUS \
                                --gpu_memory_utilization 0.9 \
                                --batch_size 8 \
                                --swap_space 8 \
                                --download_dir $HOME/.cache/huggingface/hub \
                                --verbose \
                                --resume \
                                --config_dir $backward_config \
                                --use_cache \
                                --datasets llmbar_natural llmbar_adversarial mtbench instrusum
                            break
                        fi
                    done
                done
                break
            fi
        done
    done
fi

# Run protocol-consistency, which only needs to vote on the previous results

SELECTED_MODELS=(
    llama-3-70b
    llama-3-8b
    llama-2-70b
    llama-2-13b
    llama-2-7b
    tulu-2-dpo-70b
    tulu-2-70b
    tulu-2-dpo-13b
    tulu-2-13b
    tulu-2-dpo-7b
    tulu-2-7b
    mixtral-8x7b
    yi-1.5-34b
    yi-1.5-9b
    qwen-1.5-72b
    qwen-1.5-32b
    qwen-2-72b
    gemma-7b
    gemma-2b
    mistral-7b-v0.3
    glm-4-9b
    llama-3.1-70b
    llama-3.1-8b
    qwen-2.5-72b
)

SELECTED_CONFIGS=(
    protocol-consistency
)

for config in "${SELECTED_CONFIGS[@]}"; do
    for j in "${!CONFIGS[@]}"; do
        if [[ "${CONFIGS[j]}" == "${config}" ]]; then
            for model in "${SELECTED_MODELS[@]}"; do
                for i in "${!NAMES[@]}"; do
                    if [[ "${NAMES[i]}" == "$model" ]]; then
                        # run the forward pass
                        forward_config=${CONFIG_DIRS[$j]}.yaml
                        echo ${config} $forward_config ${NAMES[$i]} ${CKPTS[$i]} ${CLSS[$i]} 
                        RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=${GPU_IDS} python run.py \
                            --model_pt ${CKPTS[$i]} \
                            --model_name ${NAMES[$i]} \
                            --model_cls ${CLSS[$i]} \
                            --tensor_parallel_size $NUM_GPUS \
                            --gpu_memory_utilization 0.9 \
                            --batch_size 8 \
                            --swap_space 8 \
                            --download_dir $HOME/.cache/huggingface/hub \
                            --verbose \
                            --resume \
                            --config_dir $forward_config \
                            --use_cache \
                            --datasets llmbar_natural llmbar_adversarial mtbench instrusum \
                            --no_model
                        # run the swap pass
                        backward_config=${CONFIG_DIRS[$j]}_swap.yaml
                        echo ${config}_swap $backward_config ${NAMES[$i]} ${CKPTS[$i]} ${CLSS[$i]} 
                        RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=${GPU_IDS} python run.py \
                            --model_pt ${CKPTS[$i]} \
                            --model_name ${NAMES[$i]} \
                            --model_cls ${CLSS[$i]} \
                            --tensor_parallel_size $NUM_GPUS \
                            --gpu_memory_utilization 0.9 \
                            --batch_size 8 \
                            --swap_space 8 \
                            --download_dir $HOME/.cache/huggingface/hub \
                            --verbose \
                            --resume \
                            --config_dir $backward_config \
                            --use_cache \
                            --datasets llmbar_natural llmbar_adversarial mtbench instrusum \
                            --no_model
                        break
                    fi
                done
            done
            break
        fi
    done
done