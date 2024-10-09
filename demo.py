from ReIFE.evaluator import PairwiseEvaluator
from ReIFE.methods import get_method, get_parser
from ReIFE.models import get_model
from ReIFE.utils import get_dataset_path
from functools import partial
from helper import model_info, method_info, parser_info
import os
from meta_eval import pairwise_meta_eval


def demo():
    # Get model info
    model_info("llama2vllm")
    # Load the model
    model_cls = get_model("llama2vllm")
    home_dir = os.getenv("HOME")
    model = model_cls(
        model_pt="meta-llama/Llama-2-7b-chat-hf",
        # below are vllm specific args
        tensor_parallel_size=1,
        download_dir=os.path.join(home_dir, ".cache/huggingface/hub"),
        gpu_memory_utilization=0.9,
        quantization=None,
        swap_space=8,
        max_input_len=4000,  # max input length
        max_model_len=4096,  # max model length
        dtype="float32",
    )
    # Load the evaluator
    evaluator = PairwiseEvaluator(model)
    # Get method info
    eval_method = "base_pairwise"
    method_info(eval_method)
    # Load the evaluation method
    eval_fn = get_method(eval_method)
    eval_fn = partial(
        eval_fn,
        # in run.py, below should be given by the config file
        instruction_marker="INSTRUCTION",
        output_marker="OUTPUT",
    )
    # Get parser info
    parser_info("base_pairwise")
    # Load the parser, in run.py, below should be given by the config file
    parse_fn = get_parser("base_pairwise")
    parse_fn = partial(
        parse_fn,
        sys1_marker="a",  # sys1 marker
        sys2_marker="b",  # sys2 marker
        pattern=r"Output \((.*?)\)",  # pattern to match the output
        verbose=True,
    )
    model_name = "llama-2-7b"
    prompt_method = "pairwise_vanilla"  # it corresponds to the prompt file in /prompts
    eval_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,  # number of samples
        "max_tokens": 64,  # max tokens
        "logprobs": 32,  # top k logprobs to return
    }
    dataset = "llmbar_natural"
    dataset_path = get_dataset_path(dataset)
    batch_size = 1
    # Run the evaluation
    evaluator.pairwise_eval(
        eval_fn=eval_fn,
        input_dir=dataset_path,
        output_dir=f"results/{dataset}.{model_name}.{eval_method}.{prompt_method}.jsonl",
        prompt_dir=f"prompts/{prompt_method}.txt",
        batch_size=batch_size,
        output_text_dir=f"results/outputs/{dataset}.{model_name}.{eval_method}.{prompt_method}.txt",
        parse_fn=parse_fn,
        verbose=True,
        **eval_kwargs,
    )
    # Run the meta-evaluation
    pairwise_meta_eval(
        human_dir=dataset_path,
        model_dir=f"results/{dataset}.{model_name}.{eval_method}.{prompt_method}.jsonl",
        verbose=True,
    )


if __name__ == "__main__":
    demo()
