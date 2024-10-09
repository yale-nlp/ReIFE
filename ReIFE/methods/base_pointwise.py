from .registry import register_parser, register_method
import numpy as np
from scipy.special import logsumexp
from ..base_llm import BaseLLM, BaseVLLM
from .utils import prompt_to_chatml, open_utf8
import json
from tqdm import tqdm
import torch


@register_parser("base_pointwise")
def base_pointwise_parse(
    data: dict, num_weighted: int = 1, base_score: int = 0, verbose: bool = False
) -> tuple[dict, bool]:
    """
    Parse the response from the model.

    Args:
        data: The data to be parsed.
        num_weighted: The number of weighted logprobs.
        verbose: Whether to print verbose output.

    Returns:
        dict: The parsed response.
    """
    logprobs = data["response"][0]["logprobs"][0]

    def is_digit(x):
        try:
            int(x)
            return True
        except:
            return False

    logprobs = {int(k): v for k, v in logprobs.items() if is_digit(k)}
    if len(logprobs) == 0:
        if verbose:
            print(
                f"Empty logprobs for {data['instruction']}: {data['response'][0]['text']}"
            )
        # score = random.randint(0, 9)
        score = base_score
        fail = True
    else:
        logprobs = [(k, v) for k, v in logprobs.items()]
        logprobs.sort(key=lambda x: x[1], reverse=True)
        logprobs = logprobs[:num_weighted]
        logprobs = {k: v for k, v in logprobs}
        scores = [k for k in logprobs.keys()]
        logprobs = [logprobs[k] for k in scores]
        scores = np.array(scores)
        logprobs = np.array(logprobs)
        norm_logprobs = logprobs - logsumexp(logprobs)
        score = np.sum(scores * np.exp(norm_logprobs)).item()
        fail = False
    return {"score": score}, fail


@register_method("base_pointwise")
def pointwise_eval(
    model: BaseLLM | BaseVLLM,
    data: list[dict],
    output_dir: str,
    prompt_dir: str,
    batch_size: int,
    instruction_marker: str,
    output_marker: str,
    output_text_dir: str | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    max_tokens: int = 512,
    logprobs: int | None = None,
    verbose: bool = False,
    **kwargs,
) -> None:
    """
    Perform pointwise evaluation.

    Args:
        model: The model for evaluation.
        data: The data to evaluate.
        output_dir: The directory to save the evaluation output.
        prompt_dir: The directory containing the prompt data.
        batch_size: The batch size for evaluation.
        instruction_marker: The marker for the instruction.
        output_marker: The marker for the output.
        output_text_dir: The directory to save the output text.
        temperature: The temperature for sampling.
        top_p: The top-p value for sampling.
        n: The number of samples to generate.
        max_tokens: The maximum number of generated tokens for each sample.
        logprobs: The number of log probabilities to output.
        parse_fn: The function to parse the output.
        verbose: Whether to print verbose output.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        None
    """
    # convert prompt to chatml
    with open_utf8(prompt_dir) as f:
        prompt = f.read().strip()
    inputs = []
    for example in data:
        kwargs = {
            instruction_marker: example["instruction"],
            output_marker: example["output"],
        }
        inputs.append(prompt_to_chatml(prompt.format_map(kwargs)))
    print(f"Generating {len(data)} examples...")
    print(f"Prompt example: {json.dumps(inputs[0], indent=2)}")

    if output_text_dir is not None:
        f_text = open_utf8(output_text_dir, "w")
    with open_utf8(output_dir, "w") as f:
        for i in tqdm(range(0, len(inputs), batch_size), desc="Generating", disable=not verbose):
            with torch.no_grad():
                outputs = model.generate(
                    inputs[i : i + batch_size],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    logprobs=logprobs,
                    use_tqdm=False,
                )
                for j, response in enumerate(outputs):
                    output = {
                        "prompt": inputs[i + j],
                        "response": response,
                        "data": data[i + j],
                    }
                    text = response[0]["text"].replace("\n", " ").strip()
                    if output_text_dir is not None:
                        print(text, file=f_text, flush=True)
                    print(json.dumps(output), file=f, flush=True)
    if output_text_dir is not None:
        f_text.close()


@register_parser("base_likelihood")
def base_likelihood_parse(data: dict, length_avg: bool = False) -> dict:
    """
    Parse the response from the model.

    Args:
        data: The data to be parsed.
        length_avg: The average length of the responses.
        verbose: Whether to print verbose output.

    Returns:
        float: The likelihood of the response.
    """
    logprobs = data["response"]
    if length_avg:
        logprobs = sum(logprobs) / len(logprobs)
    else:
        logprobs = sum(logprobs)
    return {"score": logprobs}


@register_method("base_likelihood")
def pointwise_likelihood(
    model: BaseLLM | BaseVLLM,
    data: list[dict],
    output_dir: str,
    prompt_dir: str,
    batch_size: int,
    instruction_marker: str,
    output_marker: str,
    output_score_dir: str | None = None,
    verbose: bool = False,
    **kwargs,
) -> None:
    """
    Perform likelihood evaluation.

    Args:
        model: The model for evaluation.
        data: The data to evaluate.
        output_dir: The directory to save the evaluation output.
        prompt_dir: The directory containing the prompt data.
        batch_size: The batch size for evaluation.
        instruction_marker: The marker for the instruction.
        output_marker: The marker for the output.
        output_score_dir: The directory to save the output scores.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        None
    """
    # convert prompt to chatml
    with open_utf8(prompt_dir) as f:
        prompt = f.read().strip()
    inputs = []
    for example in data:
        kwargs = {
            instruction_marker: example["instruction"],
            output_marker: example["output"],
        }
        inputs.append(prompt_to_chatml(prompt.format_map(kwargs)))
    print(f"Generating {len(data)} examples...")
    print(f"Prompt example: {json.dumps(inputs[0], indent=2)}")

    if output_score_dir is not None:
        f_score = open_utf8(output_score_dir, "w")
    with open_utf8(output_dir, "w") as f:
        for i in tqdm(range(0, len(inputs), batch_size), desc="Scoring", disable=not verbose):
            outputs = model.score(inputs[i : i + batch_size])
            for j, response in enumerate(outputs):
                output = {
                    "prompt": inputs[i + j],
                    "response": response,
                    "data": data[i + j],
                }
                if output_score_dir is not None:
                    print(json.dumps(response), file=f_score, flush=True)
                print(json.dumps(output), file=f, flush=True)
    if output_score_dir is not None:
        f_score.close()
