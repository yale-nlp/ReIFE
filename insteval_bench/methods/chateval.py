from .registry import register_method
import random
from .utils import prompt_to_chatml, open_utf8
import json
from tqdm import tqdm
from ..base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
import torch
import os


def _construct_chat_history(chat_history: list[tuple], **kwargs):
    history_str = ""
    for role, response_text in chat_history:
        history_str += f"{role}: {response_text}\nEND OF {role.capitalize()} Argument\n\n"
    return history_str.strip()


@register_method("pairwise_chateval")
def pairwise_multi_role_debate(
    model: BaseLLM | BaseVLLM | BaseLLMAPI,
    data: list[dict],
    output_dir: str,
    prompt_dir: str,
    first_message_dir: str,
    role_descriptions: list[str],
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
    num_rounds: int = 1,
    random_role_order: bool = False,
    **kwargs,
) -> None:
    """
    Perform pairwise evaluation using chateval--debate among multiple roles--method.
    Args:
        model: The model to evaluate.
        data: The data to evaluate.
        output_dir: The directory to save the evaluation output.
        prompt_dir: The directory containing the prompt data.
        first_message_dir: The directory containing the first message prompt upon the first round and the first role (index 0).
        role_descriptions: The list of directories containing the role descriptions.
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
        num_rounds: The number of rounds.
        random_role_order: Whether to use a random role order.
        kwargs: Additional keyword arguments.
    Returns:
        None
    """
    with open_utf8(first_message_dir) as f:
        initial_prompt = f.read().strip()
    with open_utf8(prompt_dir) as f:
        subsequent_prompt = f.read().strip()

    if random_role_order:
        role_order = random.sample(range(len(role_descriptions)), len(role_descriptions))
    else:
        role_order = list(range(len(role_descriptions)))

    chat_history = [[] for _ in range(len(data))]
    for round_idx in range(num_rounds):
        for role_idx in role_order:
            role_description_dir = role_descriptions[role_idx]

            with open_utf8(role_description_dir) as f:
                role_description = f.read().strip()
            role = role_description.split(",")[0].split(" ",4)[-1]
            
            inputs = []
            for i, example in enumerate(data):
                kwargs = {
                    instruction_marker: example["instruction"],
                    output_marker + "_1": example["output_1"],
                    output_marker + "_2": example["output_2"],
                    "ROLE_DESCRIPTION": role_description,
                }
                if round_idx == 0 and role_idx == 0:
                    input = initial_prompt.format_map(kwargs)
                else:
                    kwargs["CHAT_HISTORY"] = _construct_chat_history(chat_history[i])
                    input = subsequent_prompt.format_map(kwargs)
                input = prompt_to_chatml(input)
                inputs.append(input)
            if use_cache and os.path.exists(os.path.join(cache_dir, f"{round_idx}_{role}_outputs.json")):
                with open_utf8(os.path.join(cache_dir, f"{round_idx}_{role}_outputs.json")) as f:
                    outputs = json.load(f)
            else:
                if verbose:
                    print(f"Generating {len(data)} examples for role: {role}...")
                    print(f"Prompt example: {json.dumps(inputs[0], indent=2)}")

                outputs = []
                with torch.no_grad():
                    for i in tqdm(
                        range(0, len(inputs), batch_size),
                        desc=f"Generating for role: {role}",
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
                    with open_utf8(os.path.join(cache_dir, f"{role}_{round_idx}_outputs.json"), "w") as f:
                        json.dump(outputs, f, indent=2)
            for i, response in enumerate(outputs):
                chat_history[i].append((role, response[0]["text"]))

    if output_text_dir is not None:
        with open_utf8(output_text_dir, "w") as f_text:
            for response_text in chat_history:
                print("|||".join([":".join(x).replace("\n", " ").strip() for x in response_text]), file=f_text)

    with open_utf8(output_dir, "w") as f:
        for example, response, inp, chat_history in zip(data, outputs, inputs, chat_history):
            output = {
                "prompt": inp,
                "response": response,
                "chat_history": chat_history,
                "data": example,
            }
            print(json.dumps(output), file=f, flush=True)
