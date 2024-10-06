from .registry import register_method, register_parser
from transformers import AutoTokenizer
import random
import re
from .utils import prompt_to_chatml, open_utf8
import json
from tqdm import tqdm
from ..base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
import torch
from scipy.special import logsumexp
import numpy as np
import os
from collections import Counter

@register_parser("base_pairwise")
def base_pairwise_parse(
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


@register_parser("base_pairwise_cot")
def base_pairwise_cot_parse(
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


@register_parser("base_pairwise_sc")
def base_pairwise_sc_parse(
    data: dict,
    sys1_marker: str = "m",
    sys2_marker: str = "M",
    pattern: str | None = None,
    verbose: bool = False,
) -> tuple[dict, bool]:
    """
    Parse the response from the model with self-consistency.

    Args:
        data: The data to be parsed.
        sys1_marker: The marker for system 1.
        sys2_marker: The marker for system 2.
        pattern: The pattern to match the response.
        verbose: Whether to print verbose output.

    Returns:
        tuple[dict, bool]: The parsed response and whether the parsing failed.
    """
    fail = False
    tie = 0
    winners = []
    for response in data["response"]:
        result, _fail = base_pairwise_parse(
            {"response": [response]}, sys1_marker, sys2_marker, pattern, verbose
        )
        if _fail:
            fail = True
        else:
            winners.append(result["winner"])
    if len(winners) == 0:
        result = random.randint(1, 2)
    else:
        counter = Counter(winners)
        if counter[1] > counter[2]:
            result = 1
        elif counter[1] < counter[2]:
            result = 2
        else:
            result = random.randint(1, 2)
            tie = 1
    result = {"winner": result, "tie": tie, "fail": fail, "winners": winners}
    return result, fail


@register_parser("base_pairwise_logprob")
def base_pairwise_logprob_parse(
    data: dict,
    sys1_marker: str = "m",
    sys2_marker: str = "M",
    pattern: str | None = None,
    tokenizer: AutoTokenizer | None = None,
    verbose: bool = False,
) -> tuple[dict, bool]:
    """
    Parse the response from the model.

    Args:
        data: The data to be parsed.
        sys1_marker: The marker for system 1.
        sys2_marker: The marker for system 2.
        verbose: Whether to print verbose output.
        tokenizer: The tokenizer to use.
        pattern: The pattern to match the response.

    Returns:
        tuple[dict, bool]: The parsed response and whether the parsing failed.
    """
    response = data["response"][0]
    fail = False
    if pattern is None:
        # match the first occurrence of the marker
        logprobs = response["logprobs"][0]
    else:
        text = response["text"]
        match = re.search(pattern, text)
        no_match = True
        if match:
            start_index, end_index = match.span(1)
            found_token = text[start_index:end_index]
            prefix_index = len(tokenizer.tokenize(text[:start_index])) - 1
            label_index = len(tokenizer.tokenize(text[:end_index])) - 1
            if label_index - prefix_index > 1:
                # raise ValueError("More than one token in the label")
                print("Warning: More than one token in the label")
            token = response["tokens"][label_index]
            if token != found_token:
                # raise ValueError(f"Token {token} does not match found {found_token}")
                print(f"Warning: Token {token} does not match found {found_token}")
            else:
                logprobs = response["logprobs"][label_index]
                no_match = False
        if (not match) or no_match:
            # search for the first occurrence of the marker
            for i, token in enumerate(response["tokens"]):
                if token == sys1_marker or token == sys2_marker:
                    logprobs = response["logprobs"][i]
                    no_match = False
                    break
            if no_match:
                # no mathing pattern, use the first token
                logprobs = response["logprobs"][0]
                fail = True
                if verbose:
                    print(
                        f"No matching pattern for {data['instruction']}: {response['text']}"
                    )

    tokens = logprobs.keys()
    tie = 0
    if sys1_marker in tokens and sys2_marker in tokens:
        if logprobs[sys1_marker] > logprobs[sys2_marker]:
            result = 1
        elif logprobs[sys1_marker] < logprobs[sys2_marker]:
            result = 2
        else:
            result = random.randint(1, 2)
            tie = 1
    elif sys1_marker in tokens:
        result = 1
    elif sys2_marker in tokens:
        result = 2
    else:
        if verbose:
            print(f"Empty logprobs for {data['instruction']}: {response['text']}")
        result = random.randint(1, 2)
        fail = True
    result = {"winner": result, "tie": tie, "fail": fail}
    result["logprobs_1"] = logprobs[sys1_marker] if sys1_marker in tokens else None
    result["logprobs_2"] = logprobs[sys2_marker] if sys2_marker in tokens else None
    return result, fail


@register_method("base_pairwise")
def base_pairwise_eval(
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
    **kwargs,
) -> None:
    """
    Perform pairwise evaluation.

    Args:
        model: The model to evaluate.
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
        verbose: Whether to print verbose output.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        None
    """

    with open_utf8(prompt_dir) as f:
        prompt = f.read().strip()
    # convert prompt to chatml
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


@register_method("base_pairwise_raw")
def base_pairwise_raw_eval(
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
    **kwargs,
) -> None:
    """
    Perform pairwise evaluation.

    Args:
        model: The model to evaluate.
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
        verbose: Whether to print verbose output.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        None
    """

    with open_utf8(prompt_dir) as f:
        prompt = f.read().strip()
    # convert prompt to chatml
    inputs = []
    def format_key(key):
        return "{" + key + "}"
    for example in data:
        input = prompt.replace(format_key(instruction_marker), example["instruction"])
        input = input.replace(format_key(output_marker + "_1"), example["output_1"])
        input = input.replace(format_key(output_marker + "_2"), example["output_2"])
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


@register_parser("base_pairwise_bidirectional")
def base_pairwise_bidirectional_eval(
    forward_data: dict,
    backward_data: dict,
    verbose: bool = False,
    **kwargs,
) -> tuple[int, dict]:
    """
    Combine the forward and backward results.

    Args:
        forward_data: The forward data.
        backward_data: The backward data.
        verbose: Whether to print verbose output.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        tuple[int, dict]: The result and the statistics.
    """

    def normalized_logprobs(logprob1, logprob2):
        fail = False
        if logprob1 is None and logprob2 is None:
            prob = 0.5
            fail = True
        elif logprob1 is None:
            prob = 0
        elif logprob2 is None:
            prob = 1
        else:
            prob = np.exp(logprob1 - logsumexp([logprob1, logprob2])).item()
        return prob, fail

    forward_prob_1, forward_fail = normalized_logprobs(
        forward_data["logprobs_1"], forward_data["logprobs_2"]
    )
    backward_prob_1, backward_fail = normalized_logprobs(
        backward_data["logprobs_1"], backward_data["logprobs_2"]
    )
    avg_prob_1 = (forward_prob_1 + backward_prob_1) / 2
    tie = 0
    if avg_prob_1 > 0.5:
        winner = 1
    elif avg_prob_1 < 0.5:
        winner = 2
    else:
        winner = random.randint(1, 2)
        tie = 1
    forward_winner = forward_data["winner"]
    backward_winner = backward_data["winner"]
    agreement = forward_winner == backward_winner
    stats = {
        "tie": tie,
        "agreement": agreement,
        "forward_fail": forward_fail,
        "backward_fail": backward_fail,
    }
    return winner, stats


@register_method("base_pairwise_metric")
def base_pairwise_metric_eval(
    model: BaseLLM | BaseVLLM | BaseLLMAPI,
    data: list[dict],
    output_dir: str,
    prompt_dir: str,
    metric_prompt_dir: str,
    batch_size: int,
    instruction_marker: str,
    output_marker: str,
    output_text_dir: str | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    max_tokens: int = 512,
    metric_max_tokens: int = 512,
    logprobs: int | None = None,
    verbose: bool = False,
    metric_marker: str = "QUESTIONS",
    use_cache: bool = False,
    cache_dir: str | None = None,
    auxiliary_input_len: int | None = None,
    **kwargs,
) -> None:
    """
    Perform pairwise evaluation.

    Args:
        model: The model to evaluate.
        data: The data to evaluate.
        output_dir: The directory to save the evaluation output.
        prompt_dir: The directory containing the prompt data.
        metric_prompt_dir: The directory containing the metric prompt data.
        batch_size: The batch size for evaluation.
        instruction_marker: The marker for the instruction.
        output_marker: The marker for the output.
        output_text_dir: The directory to save the output text.
        temperature: The temperature for sampling.
        top_p: The top-p value for sampling.
        n: The number of samples to generate.
        max_tokens: The maximum number of generated tokens for each sample.
        metric_max_tokens: The maximum number of generated tokens for each metric.
        logprobs: The number of log probabilities to output.
        verbose: Whether to print verbose output.
        metric_marker: The marker for the metric.
        use_cache: Whether to use the cache.
        cache_dir: The directory to save the cache.
        auxiliary_input_len: The length of the auxiliary input. If None, use the default model input length.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        None
    """
    # generating metrics
    if use_cache and os.path.exists(os.path.join(cache_dir, "metrics.json")):
        with open_utf8(os.path.join(cache_dir, "metrics.json")) as f:
            metrics = json.load(f)
    else:
        if use_cache:
            print("Warning: cache not found, generating metrics...")
        with open_utf8(metric_prompt_dir) as f:
            metric_prompt = f.read().strip()
        inputs = []
        instruction_to_idx = {}
        data_idx = []
        for (i, example) in enumerate(data):
            instruction = example["instruction"]
            if instruction not in instruction_to_idx:
                instruction_to_idx[instruction] = len(inputs)
                data_idx.append(len(inputs))
                kwargs = {instruction_marker: instruction}
                input = metric_prompt.format_map(kwargs)
                input = prompt_to_chatml(input)
                inputs.append(input)
            else:
                data_idx.append(instruction_to_idx[instruction])
        
        if verbose:
            print(f"Generating {len(inputs)} metrics...")
            print(f"Prompt example: {json.dumps(inputs[0], indent=2)}")

        _metrics = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(inputs), batch_size), desc="Generating", disable=not verbose
            ):
                outputs = model.generate(
                    inputs[i : i + batch_size],
                    max_tokens=metric_max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    n=1,
                    logprobs=None,
                    use_tqdm=False,
                    input_length=auxiliary_input_len,
                )
                for j, response in enumerate(outputs):
                    metric = response[0]["text"].strip()
                    _metrics.append(metric)
        metrics = [_metrics[i] for i in data_idx]

        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open_utf8(os.path.join(cache_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

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
        input = prompt.format_map(kwargs)
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


@register_method("base_pairwise_reference")
def base_pairwise_reference_eval(
    model: BaseLLM | BaseVLLM | BaseLLMAPI,
    data: list[dict],
    output_dir: str,
    prompt_dir: str,
    reference_prompt_dir: str,
    batch_size: int,
    instruction_marker: str,
    output_marker: str,
    output_text_dir: str | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    max_tokens: int = 512,
    reference_max_tokens: int = 1024,
    logprobs: int | None = None,
    verbose: bool = False,
    reference_marker: str = "REFERENCE",
    use_cache: bool = False,
    cache_dir: str | None = None,
    auxiliary_input_len: int | None = None,
    **kwargs,
) -> None:
    """
    Perform pairwise evaluation.

    Args:
        model: The model to evaluate.
        data: The data to evaluate.
        output_dir: The directory to save the evaluation output.
        prompt_dir: The directory containing the prompt data.
        reference_prompt_dir: The directory containing the metric prompt data.
        batch_size: The batch size for evaluation.
        instruction_marker: The marker for the instruction.
        output_marker: The marker for the output.
        output_text_dir: The directory to save the output text.
        temperature: The temperature for sampling.
        top_p: The top-p value for sampling.
        n: The number of samples to generate.
        max_tokens: The maximum number of generated tokens for each sample.
        reference_max_tokens: The maximum number of generated tokens for the reference.
        logprobs: The number of log probabilities to output.
        verbose: Whether to print verbose output.
        reference_marker: The marker for the reference.
        use_cache: Whether to use the cache.
        cache_dir: The directory to save the cache.
        auxiliary_input_len: The length of the auxiliary input. If None, use the default model input length.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        None
    """
    # generating references
    if use_cache and os.path.exists(os.path.join(cache_dir, "references.json")):
        with open_utf8(os.path.join(cache_dir, "references.json")) as f:
            references = json.load(f)
    else:
        if use_cache:
            print("Warning: cache not found, generating references...")
        with open_utf8(reference_prompt_dir) as f:
            reference_prompt = f.read().strip()
        inputs = []
        instruction_to_idx = {}
        data_idx = []
        for (i, example) in enumerate(data):
            instruction = example["instruction"]
            if instruction not in instruction_to_idx:
                instruction_to_idx[instruction] = len(inputs)
                data_idx.append(len(inputs))
                kwargs = {instruction_marker: instruction}
                input = reference_prompt.format_map(kwargs)
                input = prompt_to_chatml(input)
                inputs.append(input)
            else:
                data_idx.append(instruction_to_idx[instruction])
        
        if verbose:
            print(f"Generating {len(inputs)} references...")
            print(f"Prompt example: {json.dumps(inputs[0], indent=2)}")

        _references = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(inputs), batch_size), desc="Generating", disable=not verbose
            ):
                outputs = model.generate(
                    inputs[i : i + batch_size],
                    max_tokens=reference_max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    n=1,
                    logprobs=None,
                    use_tqdm=False,
                    input_length=auxiliary_input_len,
                )
                for j, response in enumerate(outputs):
                    reference = response[0]["text"].strip()
                    _references.append(reference)
        references = [_references[i] for i in data_idx]

        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open_utf8(os.path.join(cache_dir, "references.json"), "w") as f:
                json.dump(references, f, indent=2)

    with open_utf8(prompt_dir) as f:
        prompt = f.read().strip()
    # convert prompt to chatml
    inputs = []
    for i, example in enumerate(data):
        kwargs = {
            instruction_marker: example["instruction"],
            output_marker + "_1": example["output_1"],
            output_marker + "_2": example["output_2"],
            reference_marker: references[i],
        }
        input = prompt.format_map(kwargs)
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
                        "reference": references[i + j],
                        "response": response,
                        "data": data[i + j],
                    }
                    text = response[0]["text"].replace("\n", " ").strip()
                    if output_text_dir is not None:
                        print(text, file=f_text, flush=True)
                    print(json.dumps(output), file=f, flush=True)
    if output_text_dir is not None:
        f_text.close()


@register_method("base_pairwise_metric_reference")
def base_pairwise_metric_reference_eval(
    model: BaseLLM | BaseVLLM | BaseLLMAPI,
    data: list[dict],
    output_dir: str,
    prompt_dir: str,
    metric_prompt_dir: str,
    reference_prompt_dir: str,
    batch_size: int,
    instruction_marker: str,
    output_marker: str,
    output_text_dir: str | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    n: int = 1,
    max_tokens: int = 512,
    reference_max_tokens: int = 1024,
    metric_max_tokens: int = 512,
    logprobs: int | None = None,
    verbose: bool = False,
    reference_marker: str = "REFERENCE",
    metric_marker: str = "QUESTIONS",
    use_cache: bool = False,
    cache_dir: str | None = None,
    auxiliary_input_len: int | None = None,
    **kwargs,
) -> None:
    """
    Perform pairwise evaluation.

    Args:
        model: The model to evaluate.
        data: The data to evaluate.
        output_dir: The directory to save the evaluation output.
        prompt_dir: The directory containing the prompt data.
        metric_prompt_dir: The directory containing the metric prompt data.
        reference_prompt_dir: The directory containing the metric prompt data.
        batch_size: The batch size for evaluation.
        instruction_marker: The marker for the instruction.
        output_marker: The marker for the output.
        output_text_dir: The directory to save the output text.
        temperature: The temperature for sampling.
        top_p: The top-p value for sampling.
        n: The number of samples to generate.
        max_tokens: The maximum number of generated tokens for each sample.
        reference_max_tokens: The maximum number of generated tokens for the reference.
        logprobs: The number of log probabilities to output.
        verbose: Whether to print verbose output.
        reference_marker: The marker for the reference.
        metric_marker: The marker for the metric.
        use_cache: Whether to use the cache.
        cache_dir: The directory to save the cache.
        auxiliary_input_len: The length of the auxiliary input. If None, use the default model input length.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        None
    """
    # generating metrics
    if use_cache and os.path.exists(os.path.join(cache_dir, "metrics.json")):
        with open_utf8(os.path.join(cache_dir, "metrics.json")) as f:
            metrics = json.load(f)
    else:
        if use_cache:
            print("Warning: cache not found, generating metrics...")
        with open_utf8(metric_prompt_dir) as f:
            metric_prompt = f.read().strip()
        inputs = []
        instruction_to_idx = {}
        data_idx = []
        for (i, example) in enumerate(data):
            instruction = example["instruction"]
            if instruction not in instruction_to_idx:
                instruction_to_idx[instruction] = len(inputs)
                data_idx.append(len(inputs))
                kwargs = {instruction_marker: instruction}
                input = metric_prompt.format_map(kwargs)
                input = prompt_to_chatml(input)
                inputs.append(input)
            else:
                data_idx.append(instruction_to_idx[instruction])
        
        if verbose:
            print(f"Generating {len(inputs)} metrics...")
            print(f"Prompt example: {json.dumps(inputs[0], indent=2)}")

        _metrics = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(inputs), batch_size), desc="Generating", disable=not verbose
            ):
                outputs = model.generate(
                    inputs[i : i + batch_size],
                    max_tokens=metric_max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    n=1,
                    logprobs=None,
                    use_tqdm=False,
                    input_length=auxiliary_input_len,
                )
                for j, response in enumerate(outputs):
                    metric = response[0]["text"].strip()
                    _metrics.append(metric)
        metrics = [_metrics[i] for i in data_idx]

        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open_utf8(os.path.join(cache_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

    # generating references
    if use_cache and os.path.exists(os.path.join(cache_dir, "references.json")):
        with open_utf8(os.path.join(cache_dir, "references.json")) as f:
            references = json.load(f)
    else:
        if use_cache:
            print("Warning: cache not found, generating references...")
        with open_utf8(reference_prompt_dir) as f:
            reference_prompt = f.read().strip()
        inputs = []
        instruction_to_idx = {}
        data_idx = []
        for (i, example) in enumerate(data):
            instruction = example["instruction"]
            if instruction not in instruction_to_idx:
                instruction_to_idx[instruction] = len(inputs)
                data_idx.append(len(inputs))
                kwargs = {instruction_marker: instruction}
                input = reference_prompt.format_map(kwargs)
                input = prompt_to_chatml(input)
                inputs.append(input)
            else:
                data_idx.append(instruction_to_idx[instruction])
        
        if verbose:
            print(f"Generating {len(inputs)} references...")
            print(f"Prompt example: {json.dumps(inputs[0], indent=2)}")

        _references = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(inputs), batch_size), desc="Generating", disable=not verbose
            ):
                outputs = model.generate(
                    inputs[i : i + batch_size],
                    max_tokens=reference_max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    n=1,
                    logprobs=None,
                    use_tqdm=False,
                    input_length=auxiliary_input_len,
                )
                for j, response in enumerate(outputs):
                    reference = response[0]["text"].strip()
                    _references.append(reference)
        references = [_references[i] for i in data_idx]

        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open_utf8(os.path.join(cache_dir, "references.json"), "w") as f:
                json.dump(references, f, indent=2)

    with open_utf8(prompt_dir) as f:
        prompt = f.read().strip()
    # convert prompt to chatml
    inputs = []
    for i, example in enumerate(data):
        kwargs = {
            instruction_marker: example["instruction"],
            output_marker + "_1": example["output_1"],
            output_marker + "_2": example["output_2"],
            reference_marker: references[i],
            metric_marker: metrics[i],
        }
        input = prompt.format_map(kwargs)
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
                        "reference": references[i + j],
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
