""" This module contains the Evaluator class, which is responsible for evaluating the performance of the model."""

import json
from typing import Callable, Any
import random
from .base_llm import BaseLLM, BaseVLLM, BaseLLMAPI
import json
from .utils import read_json, open_utf8

BasePairwiseEvalType = Callable[
    [
        (BaseLLM | BaseVLLM | BaseLLMAPI | None),  # model
        list[dict],  # data
        str,  # output_dir
        str,  # prompt_dir
        int,  # batch_size
        str,  # instruction_marker
        str,  # output_marker
        (str | None),  # output_text_dir (str | None)
        float,  # temperature
        float,  # top_p
        int,  # n
        int,  # max_tokens
        (int | None),  # logprobs (int | None)
        bool,  # verbose
        Any,  # **kwargs
    ],
    None,
]

BasePairwiseParseType = Callable[..., tuple[dict, bool]]


def base_pairwise_eval(
    model: BaseLLM | BaseVLLM | BaseLLMAPI | None,
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
    Perform pairwise evaluation. This function should be used as a template for other pairwise evaluation functions.

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
    raise NotImplementedError("This method is not implemented yet.")


class PairwiseEvaluator:
    def __init__(
        self,
        model: BaseLLM | BaseVLLM | BaseLLMAPI | None = None,
    ) -> None:
        """
        Initialize the Evaluator object.

        Args:
            model: The model to be evaluated. It should be an instance of BaseLLM or BaseVLLM.
        """
        self.model = model

    def pairwise_eval(
        self,
        eval_fn: BasePairwiseEvalType,
        input_dir: str,
        output_dir: str,
        prompt_dir: str,
        batch_size: int = 1,
        output_text_dir: str | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        max_tokens: int = 512,
        logprobs: int | None = None,
        parse_fn: BasePairwiseParseType | None = None,
        no_model: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[int, list[int]]:
        """
        Perform pairwise evaluation.

        Args:
            eval_fn: The evaluation function to be used.
            input_dir: The directory containing the input data. It should be a jsonl file.
            output_dir: The directory to save the evaluation output.
            prompt_dir: The directory containing the prompt data.
            batch_size: The batch size for evaluation.
            output_text_dir: The directory to save the output text.
            temperature: The temperature for sampling.
            top_p: The top-p value for sampling.
            n: The number of samples to generate.
            max_tokens: The maximum number of generated tokens for each sample.
            logprobs: The number of log probabilities to output.
            parse_fn: The function to parse the output.
            no_model: Whether a model is not needed.
            verbose: Whether to print verbose output.
            kwargs: Additional keyword arguments for the evaluation function.

        Returns:
            tuple[int, list[int]]: The number of fails and the list of winners.
        """
        data = read_json(input_dir)
        if self.model is not None:  # not in parse-only mode
            # call the evaluation function
            eval_fn(
                model=self.model,
                data=data,
                output_dir=output_dir,
                prompt_dir=prompt_dir,
                batch_size=batch_size,
                output_text_dir=output_text_dir,
                temperature=temperature,
                top_p=top_p,
                n=n,
                max_tokens=max_tokens,
                logprobs=logprobs,
                verbose=verbose,
                **kwargs,
            )
        elif no_model:  # evaluate without a model
            eval_fn(
                model=None,
                data=data,
                output_dir=output_dir,
                verbose=verbose,
                **kwargs,
            )
        if parse_fn is not None:
            with open_utf8(output_dir) as f:
                data = [json.loads(x) for x in f]
            fails = 0
            winners = []
            random.seed(42)  # for reproducibility
            for i, x in enumerate(data):
                result, fail = parse_fn(x)
                data[i]["result"] = result
                data[i]["winner"] = result["winner"]
                fails += fail
                winners.append(result["winner"])
            with open_utf8(output_dir, "w") as f:
                for x in data:
                    print(json.dumps(x), file=f)
            if verbose:
                print(f"Num fails: {fails}")
                print(f"Winners: {winners}")
        return fails, winners
