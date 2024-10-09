import random
from .registry import register_parser

@register_parser("prometheus_pairwise")
def prometheus_pairwise_parse(
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
    text = text.split(pattern)
    if len(text) == 2:
        match = text[1].strip()
    else:
        match = None
    fail = False
    if match:
        answer = match
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