import argparse
from insteval_bench.evaluator import PairwiseEvaluator
from insteval_bench.methods import get_method, get_parser
from insteval_bench.models import get_model
from insteval_bench.utils import get_dataset_path
from insteval_bench.base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer
import yaml
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pt", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_cls", type=str, required=True)
    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument(
        "--parse_only",
        action="store_true",
        help="Parse the results without running the evaluation.",
    )
    parser.add_argument(
        "--no_model", action="store_true", help="Run the evaluation without a model."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="In resume mode, skip already generated files.",
    )
    parser.add_argument("--use_cache", action="store_true")
    # VLLM args
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--download_dir", type=str, default="~/.cache/huggingface/hub")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--swap_space", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="auto")
    # LLMAPI args
    parser.add_argument("--parallel_size", type=int, default=1)
    parser.add_argument("--initial_wait_time", type=int, default=10)
    parser.add_argument("--end_wait_time", type=int, default=0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--key_path", type=str, default=None)
    parser.add_argument("--account_path", type=str, default=None)

    args = parser.parse_args()

    # Load the config
    with open(args.config_dir) as f:
        config = yaml.safe_load(f)

    if args.parse_only or args.no_model:
        evaluator = PairwiseEvaluator()
    else:
        # Load the evaluator
        model_cls = get_model(args.model_cls)
        if issubclass(model_cls, BaseVLLM):
            model = model_cls(
                model_pt=args.model_pt,
                tensor_parallel_size=args.tensor_parallel_size,
                download_dir=args.download_dir,
                gpu_memory_utilization=args.gpu_memory_utilization,
                quantization=args.quantization,
                swap_space=args.swap_space,
                max_input_len=config["max_input_len"],
                max_model_len=config["max_model_len"],
                dtype=args.dtype,
            )
        elif issubclass(model_cls, BaseLLMAPI):
            model = model_cls(
                model_pt=args.model_pt,
                parallel_size=args.parallel_size,
                initial_wait_time=args.initial_wait_time,
                end_wait_time=args.end_wait_time,
                max_retries=args.max_retries,
                key_path=args.key_path,
                account_path=args.account_path,
            )
        elif issubclass(model_cls, BaseLLM):
            raise NotImplementedError
        else:
            raise ValueError(f"Model class {model_cls} is not supported.")
        evaluator = PairwiseEvaluator(model)

    # Load the evaluation method
    eval_method = config["eval_method"]
    eval_fn = get_method(eval_method)
    eval_fn = partial(
        eval_fn,
        instruction_marker=config["instruction_marker"],
        output_marker=config["output_marker"],
    )

    # Load the parser
    if config["parse_method"] is not None:
        parse_fn = get_parser(config["parse_method"])
        if config["parse_with_tokenizer"]:
            parse_fn = partial(
                parse_fn,
                sys1_marker=config["sys1_marker"],
                sys2_marker=config["sys2_marker"],
                pattern=config["pattern"],
                tokenizer=AutoTokenizer.from_pretrained(
                    args.model_pt, trust_remote_code=True
                ),
                verbose=True,
            )
        else:
            parse_fn = partial(
                parse_fn,
                sys1_marker=config["sys1_marker"],
                sys2_marker=config["sys2_marker"],
                pattern=config["pattern"],
                verbose=True,
            )
    else:
        parse_fn = None

    model_name = args.model_name
    prompt_method = config["prompt_method"]
    datasets = args.datasets
    for dataset in tqdm(datasets):
        print(
            f"Running {dataset} with {model_name} using {eval_method} and {prompt_method}, config: {args.config_dir}"
        )
        dataset_path = get_dataset_path(dataset)
        eval_kwargs = config["eval_kwargs"]
        eval_kwargs["sys1_marker"] = config["sys1_marker"]
        eval_kwargs["sys2_marker"] = config["sys2_marker"]

        if "fdir_kwargs" in config:
            # Load the fdir kwargs
            for k in config["fdir_kwargs"]:
                fdir = config["fdir_kwargs"][k]
                if "{dataset}" in fdir and "{model}" in fdir:
                    fdir = fdir.format(dataset=dataset, model=model_name)
                elif "{dataset}" in fdir:
                    fdir = fdir.format(dataset=dataset)
                elif "{model}" in fdir:
                    fdir = fdir.format(model=model_name)
                else:
                    raise ValueError("Invalid fdir kwargs")
                eval_kwargs[k] = fdir
        if "file_ext" in config:
            output_dir = f"results/{dataset}.{model_name}.{eval_method}.{prompt_method}.{config['file_ext']}.jsonl"
            output_text_dir = f"results/outputs/{dataset}.{model_name}.{eval_method}.{prompt_method}.{config['file_ext']}.txt"
        else:
            output_dir = (
                f"results/{dataset}.{model_name}.{eval_method}.{prompt_method}.jsonl"
            )
            output_text_dir = f"results/outputs/{dataset}.{model_name}.{eval_method}.{prompt_method}.txt"
        exist = False
        if args.resume and os.path.exists(output_dir):
            with open(output_dir, encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            with open(dataset_path, encoding="utf-8") as f:
                dataset_data = json.load(f)
            if len(data) == len(dataset_data):
                print(f"f{output_dir} exists, skipping...")
                exist = True
        if args.use_cache:
            eval_kwargs["use_cache"] = True
        if not exist:
            fails, _ = evaluator.pairwise_eval(
                eval_fn=eval_fn,
                input_dir=dataset_path,
                output_dir=output_dir,
                prompt_dir=f"prompts/{prompt_method}.txt",
                batch_size=args.batch_size,
                output_text_dir=output_text_dir,
                parse_fn=parse_fn,
                verbose=args.verbose,
                no_model=args.no_model,
                **eval_kwargs,
            )
            if "file_ext" in config:
                log_dir = f"logs/{model_name}.{eval_method}.{prompt_method}.{dataset}.{config['file_ext']}.json"
            else:
                log_dir = (
                    f"logs/{model_name}.{eval_method}.{prompt_method}.{dataset}.json"
                )
            with open(log_dir, "w") as f:
                json.dump({"fails": fails}, f, indent=2)


if __name__ == "__main__":
    main()
