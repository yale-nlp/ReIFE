from .registry import register_method
from .utils import prompt_to_chatml, open_utf8
import json
from tqdm import tqdm
from ..base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
import torch


@register_method("pairwise_reference")
def pairwise_reference_eval(
    model: BaseLLM | BaseVLLM | BaseLLMAPI,
    data: list[dict],
    output_dir: str,
    prompt_dir: str,
    reference_dir: str,
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
    reference_marker: str = "REFERENCE",
    **kwargs,
) -> None:
    """
    Perform pairwise evaluation.

    Args:
        model: The model to evaluate.
        data: The data to evaluate.
        output_dir: The directory to save the evaluation output.
        prompt_dir: The directory containing the prompt data.
        reference_dir: The directory containing the reference data.
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
        reference_marker: The marker for the reference.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        None
    """
    with open_utf8(reference_dir) as f:
        references = json.load(f)

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