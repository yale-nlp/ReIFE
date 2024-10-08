# Get average performance of different base LLMs
python meta_eval.py \
    --datasets llmbar_natural llmbar_adversarial mtbench instrusum \
    --methods_dir misc/methods.yaml \
    --models_dir misc/models.yaml \
    --aggregate_methods \
    --cal_avg \
    --sorted \
    --use_cache

# Get average performance of benchmark evaluation protocols
python meta_eval.py \
    --datasets llmbar_natural llmbar_adversarial mtbench instrusum \
    --methods_dir misc/benchmark_methods.yaml \
    --models_dir misc/models.yaml \
    --aggregate_models \
    --cal_avg \
    --sorted \
    --use_cache 

# Get average performance of different evaluation protocols
python meta_eval.py \
    --datasets llmbar_natural llmbar_adversarial mtbench instrusum \
    --methods_dir misc/methods.yaml \
    --models_dir misc/models.yaml \
    --aggregate_models \
    --cal_avg \
    --sorted \
    --use_cache

# Get optimal performance of different evaluation protocols
python meta_eval.py \
    --datasets llmbar_natural llmbar_adversarial mtbench instrusum \
    --methods_dir misc/methods.yaml \
    --models_dir misc/models.yaml \
    --aggregate_models \
    --aggregation_mode max \
    --cal_avg \
    --sorted \
    --use_cache