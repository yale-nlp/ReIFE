import unittest

from .dummy_model import DummyAPI
from insteval_bench.evaluator import PairwiseEvaluator
from insteval_bench.methods import get_method, get_parser
from insteval_bench.utils import get_dataset_path
from functools import partial
import os
import yaml

class TestMethods(unittest.TestCase):
    def test_multimodel_synthesize(self):
        # Load the model
        model_cls = DummyAPI

        def rel_to_abs_path(rel_path):
            return os.path.join(os.path.dirname(os.path.dirname(__file__)), rel_path)
        
        config_dir = rel_to_abs_path("configs/tulu2/pairwise_synthesize_multi.yaml")
        
        with open(config_dir, "r") as f:
            config = yaml.safe_load(f)
        
        model = model_cls(
            model_pt="tmp",
            parallel_size=1,
            initial_wait_time=0,
            end_wait_time=0,
            max_retries=0,
            key_path=None,
            account_path=None,
        )
        
        # Load the evaluator
        evaluator = PairwiseEvaluator(model)

        eval_method = config["eval_method"]

        eval_fn = get_method(eval_method)
        eval_fn = partial(
            eval_fn,
            instruction_marker=config["instruction_marker"],
            output_marker=config["output_marker"],
        )

        # Load the parser
        if config["parse_method"] is not None:
            parse_fn = get_parser(config["parse_method"])

            parse_fn = partial(
                parse_fn,
                sys1_marker=config["sys1_marker"],
                sys2_marker=config["sys2_marker"],
                pattern=config["pattern"],
                verbose=True,
            )

        else:
            parse_fn = None

        model_name = "tmp"
        prompt_method = config["prompt_method"]

        dataset_path = "llmbar_natural"

        dataset = rel_to_abs_path(get_dataset_path(dataset_path))

        eval_kwargs = config["eval_kwargs"]
        eval_kwargs["sys1_marker"] = config["sys1_marker"]
        eval_kwargs["sys2_marker"] = config["sys2_marker"]

        if "fdir_kwargs" in config:
            # Reference-aided pairwise evaluation
            for k in config["fdir_kwargs"]:
                fdir = config["fdir_kwargs"][k]
                if "{dataset}" in fdir and "{model}" in fdir:
                    fdir = fdir.format(dataset=dataset, model=model_name)
                elif "{dataset}" in fdir:
                    fdir = fdir.format(dataset=dataset)
                elif "{model}" in fdir:
                    fdir = fdir.format(model=model_name)
                else:
                    raise ValueError("Invalid fdir kwargs")
                eval_kwargs[k] = fdir
        fails, _ = evaluator.pairwise_eval(
            eval_fn=eval_fn,
            input_dir=dataset_path,
            output_dir=rel_to_abs_path(f"tmp/{dataset}.{model_name}.{eval_method}.{prompt_method}.jsonl"),
            prompt_dir=rel_to_abs_path(f"prompts/{prompt_method}.txt"),
            batch_size=1,
            output_text_dir=rel_to_abs_path(f"tmp/{dataset}.{model_name}.{eval_method}.{prompt_method}.txt"),
            parse_fn=parse_fn,
            verbose=True,
            **eval_kwargs,
        )