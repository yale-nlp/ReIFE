from .registry import register_method
from ..base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
from .utils import prompt_to_chatml, open_utf8
from ..utils import read_json
import torch
import json
from tqdm import tqdm


@register_method("multi_synthesize")
def synthesize_eval_multi(
    model: BaseLLM | BaseVLLM | BaseLLMAPI,
    data: list[dict],
    output_dir: str,
    prompt_dir: str,
    batch_size: int,
    instruction_marker: str,
    explanation_marker: str,
    output_marker: str,
    sys1_marker: str,
    sys2_marker: str,
    num_assistants: int,
    prompt_util_dir: str,
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
    Synthesize the results of multiple evaluators.

    Args:
        model: The model to evaluate.
        data: The data to evaluate.
        output_dir: The directory to save the evaluation output.
        prompt_dir: The directory containing the prompt data.
        batch_size: The batch size for evaluation.
        instruction_marker: The marker for the instruction.
        explanation_marker: The marker for the explanation.
        output_marker: The marker for the output.
        sys1_marker: The marker for system 1.
        sys2_marker: The marker for system 2.
        num_assistants: The number of assistants to synthesize.
        prompt_util_dir: The directory containing the nested prompts.
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

    
    evaluators_results = []
    for i in range(num_assistants):
        evaluator_dir = kwargs[f"evaluator{i+1}_dir"]
        evaluator_results = read_json(evaluator_dir)
        evaluators_results.append(evaluator_results)
        assert len(evaluator_results) == len(data)
    with open_utf8(prompt_util_dir) as f:
        util_prompt = f.read().strip()

    # convert prompt to chatml
    inputs = []
    records = []
    for i in range(len(data)):

        winners = []
        explanations = []
        for j in range(num_assistants):
            evaluator_winner = evaluators_results[j][i]["winner"]
            assert evaluator_winner in [1, 2]
            evaluator_explanation = evaluators_results[j][i]["response"][0]["text"]
            winners.append(evaluator_winner)
            explanations.append(evaluator_explanation)

        if len(winners) == 1:
            records.append({"winner": winners[0], "no_debate": True})
        else:
            _explanations = []
            for j in range(num_assistants):
                _util_prompt = util_prompt
                _util_prompt = _util_prompt.format_map(
                    {
                        "ID": j + 1,
                        "PREF_MARKER": sys1_marker if winners[j] == 1 else sys2_marker,
                        "EXPLANATION": explanations[j],
                    }
                )
                _explanations.append(_util_prompt)

            _explanations = "\n\n".join(_explanations)
                        
            kwargs = {
                instruction_marker: data[i]["instruction"],
                output_marker + "_1": data[i]["output_1"],
                output_marker + "_2": data[i]["output_2"],
                explanation_marker: _explanations,
                "NUM": num_assistants,
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
