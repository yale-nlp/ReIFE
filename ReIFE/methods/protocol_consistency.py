from .registry import register_method, register_parser
from .utils import open_utf8
from ..utils import read_json
import json
import random
from collections import Counter

@register_parser("majority_vote")
def base_pairwise_sc_parse(
    data: dict,
    sys1_marker: str | None = None,
    sys2_marker: str | None = None,
    pattern: str | None = None,
    verbose: bool = False,
) -> tuple[dict, bool]:
    """
    Parse the response from the model with self-consistency.

    Args:
        data: The data to be parsed.
        sys1_marker: The marker for system 1. Placeholder.
        sys2_marker: The marker for system 2. Placeholder.
        pattern: The pattern to match the response. Placeholder.
        verbose: Whether to print verbose output. Placeholder.

    Returns:
        tuple[dict, bool]: The parsed response and whether the parsing failed.
    """
    fail = False
    tie = 0
    winners = [x["winner"] for x in data["response"]]
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


@register_method("protocol_vote")
def synthesize_eval_multi(
    model: None,
    data: list[dict],
    output_dir: str,
    num_assistants: int,
    verbose: bool = False,
    **kwargs,
) -> None:
    """
    Protocol vote synthesis.

    Args:
        model: Placeholder for the model. Not used.
        data: The data to evaluate.
        output_dir: The directory to save the evaluation output.
        num_assistants: The number of assistants to synthesize.
        verbose: Whether to print verbose output.
        kwargs: Additional keyword arguments for the evaluation function.

    Returns:
        None
    """    
    evaluators_results = []
    for i in range(num_assistants):
        evaluator_dir = kwargs[f"evaluator{i+1}_dir"]
        evaluator_results = read_json(evaluator_dir)
        evaluators_results.append(evaluator_results)
        assert len(evaluator_results) == len(data)

    with open_utf8(output_dir, "w") as f:
        for i in range(len(data)):
            response = [{"winner": x[i]["winner"]} for x in evaluators_results]
            print(json.dumps({"response": response, **data[i]}), file=f, flush=True)