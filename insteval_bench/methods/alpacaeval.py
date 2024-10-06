from .registry import register_parser
import re
import random

@register_parser("alpacaeval_api")
def alpacaeval_api_parse(
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
    matched = False
    fail = False
    if match:
        answer = match.group(1)
        if answer == sys1_marker:
            result = 1
            matched = True
        elif answer == sys2_marker:
            result = 2
            matched = True
        else:
            result = random.randint(1, 2)
            fail = True
            if verbose:
                print(f"Invalid answer {answer}: {text}")
    if not match or not matched:
        first_token = text.split()[0]
        if first_token == sys1_marker:
            result = 1
        elif first_token == sys2_marker:
            result = 2
        else:
            fail = True
            result = random.randint(1, 2)
            if verbose:
                print(f"No matching pattern: {text}")
    result = {"winner": result, "tie": 0, "fail": fail}
    return result, fail