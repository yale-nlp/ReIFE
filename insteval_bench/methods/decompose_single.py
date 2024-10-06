from .registry import register_parser, register_method
import re
import random
from ..base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
from .utils import prompt_to_chatml, open_utf8
import torch
import json
from tqdm import tqdm

@register_method("decomp_single")
def decomposition_synthesize_eval(
    model: BaseLLM | BaseVLLM | BaseLLMAPI,
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
) -> None:
    """
    Evaluate the model using the decomposition method.

    Args:
        model: The model to be evaluated.
        data: The data to be evaluated.
        output_dir: The output directory to save the results.
        prompt_dir: The prompt directory.
        batch_size: The batch size.
        instruction_marker: The instruction marker.
        output_marker: The output marker.
        output_text_dir: The output text directory.
        temperature: The temperature.
        top_p: The top p.
        n: The number of samples.
        max_tokens: The maximum number of tokens.
        logprobs: The number of logprobs.
        verbose: Whether to print verbose output.
    """
    with open_utf8(prompt_dir) as f:
        prompt = f.read().strip()

    inputs = []
    for example in data:
        kwargs = {
            instruction_marker: example["instruction"],
            output_marker + "_1": example["output_1"],
            output_marker + "_2": example["output_2"],
        }
        input = prompt.format_map(kwargs)
        input = prompt_to_chatml(input)
        inputs.append(input)

    print(f"Generating {len(data)} examples...")
    print(f"Prompt example: {json.dumps(inputs[0], indent=2)}")

    if output_text_dir is not None:
        f_text = open_utf8(output_text_dir, "w")
    with open_utf8(output_dir, "w") as f:
        with torch.no_grad():
            for i in tqdm(
                range(0, len(inputs), batch_size),
                desc="Generating",
                disable=not verbose,
            ):
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
                    }
                    text = response[0]["text"].replace("\n", " ").strip()
                    if output_text_dir is not None:
                        print(text, file=f_text, flush=True)
                    output.update(**data[i + j])
                    print(json.dumps(output), file=f, flush=True)
    if output_text_dir is not None:
        f_text.close()

@register_parser("decomp_single_offload")
def decomposition_synthesize_parse(
    data: dict,
    sys1_marker: str = "Baseline",
    sys2_marker: str = "Candidate",
    pattern: str | None = None,
    verbose: bool = False,
) -> tuple[dict, bool]:
    """
    Parse the response from the model.

    Args:
        data: The data to be parsed.
        sys1_marker: The marker for system 1.
        sys2_marker: The marker for system 2.
        pattern: The pattern to match the response.
        verbose: Whether to print verbose output.

    Returns:
        tuple[dict, bool]: The parsed response and whether the parsing failed.
    """
    
    response = data["response"][0]
    text = response["text"]
    matches = re.findall(pattern, text)
    fail = False
    is_swap = sys1_marker == "Candidate"
    if matches:
        scores = list(map(int, matches))
        sum_score = sum(scores)
        if sum_score > 0:
            winner = 2  # Candidate i.e. (b)
        elif sum_score < 0:
            winner = 1  # Baseline i.e. (a)
        else:
            winner = random.randint(1, 2)
            fail = True
            if verbose:
                print(f"Sum is zero, cannot determine winner: {text}")
    else:
        fail = True
        winner = random.randint(1, 2)
        scores = []
        sum_score = 0
        if verbose:
            print(f"Invalid format, cannot parse scores: {text}")
    if is_swap:
        winner = 3 - winner

    result = {
        "winner": winner,
        "scores": scores,
        "sum_score": sum_score,
        "fail": fail,
    }
    return result, fail