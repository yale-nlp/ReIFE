from .registry import register_method, register_parser
from .utils import prompt_to_chatml, open_utf8, prompt_format_map
import json
from tqdm import tqdm
from ..base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
import torch
import re
import random

@register_method("pairwise_wildbench")
def pairwise_wildbench_eval(
    model: BaseLLM | BaseVLLM | BaseLLMAPI,
    data: list[dict],
    output_dir: str,
    prompt_dir: str,
    metric_dir: str,
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
    metric_marker: str = "METRIC",
    **kwargs,
) -> None:
    """
    Perform pairwise evaluation.

    Args:
        model: The model to evaluate.
        data: The data to evaluate.
        output_dir: The directory to save the evaluation output.
        prompt_dir: The directory containing the prompt data.
        metric_dir: The directory containing the metric data.
        batch_size: The batch size for evaluation.
        instruction_marker: The marker for the instruction.
        output_marker: The marker for the output.
        output_text_dir: The directory to save the output text.
        temperature: The temperature for sampling.
        top_p: The top-p value for sampling.
        n: The number of samples to generate.
        max_tokens: The maximum number of generated tokens for each sample.
        logprobs: The number of log probabilities to output.
        verbose: Whether to print verbose output.
        metric_marker: The marker for the metric.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        None
    """
    with open_utf8(metric_dir) as f:
        metrics = json.load(f)

    with open_utf8(prompt_dir) as f:
        prompt = f.read().strip()
    # convert prompt to chatml
    inputs = []
    for i, example in enumerate(data):
        kwargs = {
            instruction_marker: example["instruction"],
            output_marker + "_1": example["output_1"],
            output_marker + "_2": example["output_2"],
            metric_marker: metrics[i],
        }
        input = prompt_format_map(prompt, kwargs)
        input = prompt_to_chatml(input)
        inputs.append(input)

    if verbose:
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
                        "metric": metrics[i + j],
                        "response": response,
                        "data": data[i + j],
                    }
                    text = response[0]["text"].replace("\n", " ").strip()
                    if output_text_dir is not None:
                        print(text, file=f_text, flush=True)
                    print(json.dumps(output), file=f, flush=True)
    if output_text_dir is not None:
        f_text.close()

def extract_values_from_json(json_string, keys = ["score", "strengths", "weaknesses", "choice"], allow_no_quotes = False):
    extracted_values = {}
    for key in keys:
        if key not in json_string:
            continue
        # Create a regular expression pattern to find the value for the given key
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            # Handle the case where the value might contain broken quotes
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
        if not match and allow_no_quotes:
            # to allow no quotes on the values
            pattern = f'"{key}"\\s*:\\s*([^,\\s]*)'
            match = re.search(pattern, json_string)
            if match:
                extracted_values[key] = match.group(1)
            else:
                # to allow no quotes on the keys
                pattern = f'{key}\\s*:\\s*([^,\\s]*)'
                match = re.search(pattern, json_string)
                if match:
                    extracted_values[key] = match.group(1)
    return extracted_values


@register_parser("pairwise_wildbench")
def pairwise_wildbench_parse(
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
        pattern: The pattern to match the response.
        verbose: Whether to print verbose output.

    Returns:
        tuple[dict, bool]: The parsed response and whether the parsing failed.
    """
    response = data["response"][0]
    text = response["text"].strip()
    # try formatting the response as json
    try:
        try:
            data = json.loads(text)
        except:
            data = extract_values_from_json(text)
    except:
        data = dict()
    if "choice" in data:
        try:
            text = str(data["choice"])
        except:
            text = ""
    else:
        text = ""
    matches = re.findall(pattern, text)
    if matches:
        answers = matches[-1]  # Take the last match
        fail = True
        if isinstance(answers, str):
            answers = [answers]
        for answer in answers:
            if answer == sys1_marker:
                result = 1
                fail = False
                break
            elif answer == sys2_marker:
                result = 2
                fail = False
                break
        if fail:
            result = random.randint(1, 2)
            if verbose:
                print(f"Invalid answer {answer}: {text} Matched: {matches}")
    else:
        if len(text.strip()) > 0 and sys1_marker == text.strip()[0]:
            result = 1
            fail = False
        elif len(text.strip()) > 0 and sys2_marker == text.strip()[0]:
            result = 2
            fail = False
        else:
            fail = True
            result = random.randint(1, 2)
            if verbose:
                print(f"No matching pattern: {text} Matched: {matches}")
    result = {"winner": result, "tie": 0, "fail": fail}
    return result, fail