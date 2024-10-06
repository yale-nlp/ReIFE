from .registry import register_method, register_parser
from ..base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
from .utils import prompt_to_chatml, open_utf8
from ..utils import read_json
import torch
import json
from tqdm import tqdm
import random
import re


@register_parser("synthesize")
def synthesize_parse(
    data: dict,
    sys1_marker: str = "m",
    sys2_marker: str = "M",
    pattern: str | None = None,
    verbose: bool = False,
) -> tuple[dict, bool]:
    """
    Parse the response from the model.

    Args:
        data: The data to be parsed.
        sys1_marker: The marker for system 1.
        sys2_marker: The marker for system 2.
        verbose: Whether to print verbose output.
        pattern: The pattern to match the response.

    Returns:
        dict: The parsed response.
    """
    response = data["response"][0]
    if "winner" in response:
        # no debate
        return {"winner": response["winner"], "tie": 0, "fail": False}, False
    text = response["text"]
    match = re.search(pattern, text)
    fail = False
    if match:
        answer = match.group(1)
        if answer == sys1_marker:
            result = 1
        elif answer == sys2_marker:
            result = 2
        else:
            result = random.randint(1, 2)
            fail = True
            if verbose:
                print(f"Invalid answer {answer}: {text}")
    else:
        fail = True
        result = random.randint(1, 2)
        if verbose:
            print(f"No matching pattern: {text}")
    result = {"winner": result, "tie": 0, "fail": fail}
    return result, fail


@register_parser("synthesize_cot")
def synthesize_cot_parse(
    data: dict,
    sys1_marker: str = "m",
    sys2_marker: str = "M",
    pattern: str | None = None,
    verbose: bool = False,
) -> tuple[dict, bool]:
    """
    Parse the response from the model.

    Args:
        data: The data to be parsed.
        sys1_marker: The marker for system 1.
        sys2_marker: The marker for system 2.
        verbose: Whether to print verbose output.
        pattern: The pattern to match the response.

    Returns:
        dict: The parsed response.
    """
    response = data["response"][0]
    if "winner" in response:
        # no debate
        return {"winner": response["winner"], "tie": 0, "fail": False}, False
    text = response["text"]
    matches = re.findall(pattern, text)
    fail = False
    if matches:
        answer = matches[-1]  # Take the last match
        if answer == sys1_marker:
            result = 1
        elif answer == sys2_marker:
            result = 2
        else:
            result = random.randint(1, 2)
            fail = True
            if verbose:
                print(f"Invalid answer {answer}: {text}")
    else:
        fail = True
        result = random.randint(1, 2)
        if verbose:
            print(f"No matching pattern: {text}")
    result = {"winner": result, "tie": 0, "fail": fail}
    return result, fail


@register_method("synthesize")
def synthesize_eval(
    model: BaseLLM | BaseVLLM | BaseLLMAPI,
    data: list[dict],
    evaluator1_dir: str,
    evaluator2_dir: str,
    output_dir: str,
    prompt_dir: str,
    batch_size: int,
    instruction_marker: str,
    explanation_marker: str,
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
    Synthesize the results of two evaluators.

    Args:
        model: The model to evaluate.
        data: The data to evaluate.
        evaluator1_dir: The directory containing the responses from the first evaluator.
        evaluator2_dir: The directory containing the responses from the second evaluator.
        output_dir: The directory to save the evaluation output.
        prompt_dir: The directory containing the prompt data.
        batch_size: The batch size for evaluation.
        instruction_marker: The marker for the instruction.
        explanation_marker: The marker for the explanation.
        output_marker: The marker for the output.
        output_text_dir: The directory to save the output text.
        temperature: The temperature for sampling.
        top_p: The top-p value for sampling.
        n: The number of samples to generate.
        max_tokens: The maximum number of generated tokens for each sample.
        logprobs: The number of log probabilities to output.
        verbose: Whether to print verbose output.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        None
    """

    with open_utf8(prompt_dir) as f:
        prompt = f.read().strip()
    evaluator1_results = read_json(evaluator1_dir)
    evaluator2_results = read_json(evaluator2_dir)
    assert len(evaluator1_results) == len(evaluator2_results) == len(data)
    # convert prompt to chatml
    inputs = []
    records = []
    for i in range(len(data)):
        evaluator1_winner = evaluator1_results[i]["winner"]
        evaluator2_winner = evaluator2_results[i]["winner"]
        assert evaluator1_winner in [1, 2]
        assert evaluator2_winner in [1, 2]
        if evaluator1_winner == evaluator2_winner:
            records.append({"winner": evaluator1_winner, "no_debate": True})
        else:
            example = data[i]
            if evaluator1_winner == 1:
                explanation1 = evaluator1_results[i]["response"][0]["text"]
                explanation2 = evaluator2_results[i]["response"][0]["text"]
            else:
                explanation1 = evaluator2_results[i]["response"][0]["text"]
                explanation2 = evaluator1_results[i]["response"][0]["text"]
            kwargs = {
                instruction_marker: example["instruction"],
                output_marker + "_1": example["output_1"],
                output_marker + "_2": example["output_2"],
                explanation_marker + "_1": explanation1,
                explanation_marker + "_2": explanation2,
            }
            input = prompt.format_map(kwargs)
            input = prompt_to_chatml(input)
            inputs.append(input)
            records.append({"no_debate": False})
    print(f"Generating {len(inputs)} examples...")
    if len(inputs) > 0:
        print(f"Prompt example: {json.dumps(inputs[0], indent=2)}")

    if output_text_dir is not None:
        f_text = open_utf8(output_text_dir, "w")
    cnt = 0
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
                j = 0
                while j < len(outputs):
                    if records[cnt]["no_debate"]:
                        output = {
                            "prompt": None,
                            "response": [{"winner": records[cnt]["winner"]}],
                        }
                        text = "No debate"
                    else:
                        response = outputs[j]
                        output = {
                            "prompt": inputs[i + j],
                            "response": response,
                        }
                        text = response[0]["text"].replace("\n", " ").strip()
                        j += 1
                    if output_text_dir is not None:
                        print(text, file=f_text, flush=True)
                    output.update(**data[cnt])
                    print(json.dumps(output), file=f, flush=True)
                    cnt += 1
        while cnt < len(data):
            output = {
                "prompt": None,
                "response": [{"winner": records[cnt]["winner"]}],
            }
            if output_text_dir is not None:
                print("No debate", file=f_text, flush=True)
            output.update(**data[cnt])
            print(json.dumps(output), file=f, flush=True)
            cnt += 1
    if output_text_dir is not None:
        f_text.close()
