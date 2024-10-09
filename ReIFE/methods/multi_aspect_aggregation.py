import json
import os
from typing import List
from .registry import register_method
from .utils import prompt_to_chatml, open_utf8
from ..base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
from tqdm import tqdm
import torch

def _construct_aspect_evaluation_history(aspect_evaluations: List[tuple], **kwargs):
    history_str = ""
    for aspect, response_text in aspect_evaluations:
        history_str += f"## Aspect: {aspect}\n## Analysis: {response_text}\n\n"
    return history_str.strip()

@register_method("pairwise_multi_aspect")
def pairwise_multi_aspect_eval(
    model: BaseLLM | BaseVLLM | BaseLLMAPI,
    data: List[dict],
    output_dir: str,
    aspect_analyze_prompt_dir: str,
    prompt_dir: str,
    aspect_descriptions: List[str],
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
    use_cache: bool = False,
    cache_dir: str | None = None,
    **kwargs,
) -> None:
    """
    Perform pairwise evaluation using chateval--debate among multiple roles--method.
    Args:
        model: The model to evaluate.
        data: The data to evaluate.
        output_dir: The directory to save the evaluation output.
        prompt_dir: The directory containing the prompt data.
        aspect_analyze_prompt_dir: The directory containing the multi-take aspect analysis prompt.
        aspect_descriptions: The list of directories containing the aspect/criteria descriptions.
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
        use_cache: Whether to use the cache.
        cache_dir: The directory to save the cache.
        **kwargs: Additional keyword arguments.
    Returns:
        None
    """

    with open_utf8(aspect_analyze_prompt_dir) as f:
        aspect_task_prompt = f.read().strip()
    with open_utf8(prompt_dir) as f:
        aggregation_prompt = f.read().strip()

    aspect_evaluations = [[] for _ in range(len(data))]

    for aspect_description_dir in aspect_descriptions:
        with open_utf8(aspect_description_dir) as f:
            aspect_description = f.read().strip()
        aspect = aspect_description.split("\n")[0].strip()

        inputs = []
        for example in data:
            kwargs = {
                instruction_marker: example["instruction"],
                output_marker + "_1": example["output_1"],
                output_marker + "_2": example["output_2"],
                "CRITERIA_TEXT": aspect_description,
            }
            input = aspect_task_prompt.format_map(kwargs)
            input = prompt_to_chatml(input)
            inputs.append(input)
        if verbose:
            print(f"Generating {len(data)} examples for aspect: {aspect}...")
            print(f"Prompt example: {json.dumps(inputs[0], indent=2)}")

        valid_cache = False
        if use_cache and os.path.exists(os.path.join(cache_dir, f"{aspect}_outputs.json")):
            with open_utf8(os.path.join(cache_dir, f"{aspect}_outputs.json")) as f:
                outputs = json.load(f)
                valid_cache = len(outputs) == len(data)
        if not valid_cache:
            outputs = []
            with torch.no_grad():
                for i in tqdm(
                    range(0, len(inputs), batch_size),
                    desc=f"Generating for aspect: {aspect}",
                    disable=not verbose,
                ):
                    batch_outputs = model.generate(
                        inputs[i : i + batch_size],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=n,
                        logprobs=logprobs,
                        use_tqdm=False,
                    )
                    outputs.extend(batch_outputs)

            if cache_dir is not None:
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                with open_utf8(os.path.join(cache_dir, f"{aspect}_outputs.json"), "w") as f:
                    json.dump(outputs, f, indent=2)

        for i, response in enumerate(outputs):
            aspect_evaluations[i].append((aspect, response[0]["text"]))

    final_inputs = []
    for i, example in enumerate(data):
        kwargs = {
            instruction_marker: example["instruction"],
            output_marker + "_1": example["output_1"],
            output_marker + "_2": example["output_2"],
            "ANALYSIS_HISTORY": _construct_aspect_evaluation_history(aspect_evaluations[i]),
        }
        input = aggregation_prompt.format_map(kwargs)
        input = prompt_to_chatml(input)
        final_inputs.append(input)

    if verbose:
        print(f"Generating {len(data)} final decisions...")
        print(f"Prompt example: {json.dumps(final_inputs[0], indent=2)}")

    final_outputs = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(final_inputs), batch_size),
            desc="Generating final decisions",
            disable=not verbose,
        ):
            batch_outputs = model.generate(
                final_inputs[i : i + batch_size],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                logprobs=logprobs,
                use_tqdm=False,
            )
            final_outputs.extend(batch_outputs)

    if output_text_dir is not None:
        with open_utf8(output_text_dir, "w") as f_text:
            for response_text in final_outputs:
                print(response_text[0]["text"].replace("\n", " ").strip(), file=f_text)

    with open_utf8(output_dir, "w") as f:
        for example, aspect_evaluation, final_input, final_output in zip(data, aspect_evaluations, final_inputs, final_outputs):
            output = {
                "aspect_evaluations": aspect_evaluation,
                "final_prompt": final_input,
                "response": final_output,
                "data": example,
            }
            print(json.dumps(output), file=f, flush=True)

@register_method("pairwise_multi_aspect_single_stage")
def pairwise_multi_aspect_single_stage_eval(
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