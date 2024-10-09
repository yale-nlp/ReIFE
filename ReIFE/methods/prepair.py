import json
import os
from typing import List
from .registry import register_method
from .utils import prompt_to_chatml, open_utf8
from ..base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
from tqdm import tqdm
import torch

def _construct_per_output_analyses(analyses: List[tuple], **kwargs):
    history_str = ""
    for output_label, analysis in analyses:
        history_str += f"Analysis for Output {output_label}:\n{analysis}\n\n"
    return history_str.strip()

@register_method("pairwise_prepair")
def pairwise_prepair_eval(
    model: BaseLLM | BaseVLLM | BaseLLMAPI,
    data: List[dict],
    output_dir: str,
    individual_analysis_prompt_dir: str,
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
    use_cache: bool = False,
    cache_dir: str | None = None,
    a_then_b: bool = True,
    **kwargs,
) -> None:
    with open_utf8(individual_analysis_prompt_dir) as f:
        individual_analysis_prompt = f.read().strip()
    with open_utf8(prompt_dir) as f:
        final_comparison_prompt = f.read().strip()

    per_output_analyses = [[] for _ in range(len(data))]

    for output_index in [0, 1]:
        output_label = 'a' if output_index == 0 else 'b'
        cache_file = f"prepair_all_output_{output_label}_analysis.json"

        if use_cache and os.path.exists(os.path.join(cache_dir, cache_file)):
            with open_utf8(os.path.join(cache_dir, cache_file)) as f:
                outputs = json.load(f)
        else:
            inputs = []
            for example in data:
                kwargs = {
                    instruction_marker: example["instruction"],
                    output_marker: example[f"output_{output_index + 1}"],
                }
                input = individual_analysis_prompt.format_map(kwargs)
                input = prompt_to_chatml(input)
                inputs.append(input)

            if verbose:
                print(f"Generating {len(data)} analyses for Output {output_label}...")
                print(f"Prompt example: {json.dumps(inputs[0], indent=2)}")

            outputs = []
            with torch.no_grad():
                for i in tqdm(
                    range(0, len(inputs), batch_size),
                    desc=f"Generating analysis for Output {output_label}",
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
                with open_utf8(os.path.join(cache_dir, cache_file), "w") as f:
                    json.dump(outputs, f, indent=2)

        for i, response in enumerate(outputs):
            if a_then_b:
                per_output_analyses[i].append((output_label, response[0]["text"]))
            else:
                per_output_analyses[i].insert(0, (output_label, response[0]["text"]))


    final_inputs = []
    for i, example in enumerate(data):
        kwargs = {
            instruction_marker: example["instruction"],
            output_marker + "_1": example["output_1"],
            output_marker + "_2": example["output_2"],
            "PER_OUTPUT_ANALYSES": _construct_per_output_analyses(per_output_analyses[i]),
        }
        input = final_comparison_prompt.format_map(kwargs)
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
        for example, per_output_analysis, final_input, final_output in zip(data, per_output_analyses, final_inputs, final_outputs):
            output = {
                "per_output_analyses": per_output_analysis,
                "final_prompt": final_input,
                "response": final_output,
                "data": example,
            }
            print(json.dumps(output), file=f, flush=True)