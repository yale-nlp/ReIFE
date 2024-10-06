from insteval_bench.utils import read_json, get_dataset_path
import argparse
from collections import defaultdict
import numpy as np
from tabulate import tabulate
import yaml
import krippendorff
from collections import Counter
import matplotlib.pyplot as plt
import json
import os
import copy
import matplotlib
import seaborn as sns
import pandas as pd
from scipy import stats
import math


def highlight(table, axis=None, skip_header=True, is_max=True):
    def highlighting(x):
        GREEN = "\033[92m"
        RESET = "\033[0m"
        return GREEN + str(x) + RESET

    if skip_header:
        header = table[0]
        table = table[1:]
    if axis == "row":
        for i, row in enumerate(table):
            row_values = [float(val) for val in row[1:]]
            if is_max:
                max_val = max(row_values)
                ids = [i + 1 for i, val in enumerate(row_values) if val == max_val]
            else:
                min_val = min(row_values)
                ids = [i + 1 for i, val in enumerate(row_values) if val == min_val]
            for id in ids:
                table[i][id] = highlighting(table[i][id])
    elif axis == "column":
        for i in range(1, len(table[0])):
            col_values = [float(row[i]) for row in table]
            if is_max:
                max_val = max(col_values)
                ids = [i for i, val in enumerate(col_values) if val == max_val]
            else:
                min_val = min(col_values)
                ids = [i for i, val in enumerate(col_values) if val == min_val]
            for id in ids:
                table[id][i] = highlighting(table[id][i])
    elif axis is None:
        table_values = [[float(val) for val in row[1:]] for row in table]
        if is_max:
            max_val = max([max(row) for row in table_values])
            ids = [
                (i, j + 1)
                for i, row in enumerate(table_values)
                for j, val in enumerate(row)
                if val == max_val
            ]
        else:
            min_val = min([min(row) for row in table_values])
            ids = [
                (i, j + 1)
                for i, row in enumerate(table_values)
                for j, val in enumerate(row)
                if val == min_val
            ]
        for i, j in ids:
            table[i][j] = highlighting(table[i][j])
    else:
        raise ValueError("Invalid axis, it must be either 'row', 'column', or None")
    if skip_header:
        table = [header] + table
    return table


def latex_highlight(table, axis=None, skip_header=True, is_max=True):
    table = copy.deepcopy(table)

    def float_fmt(x):
        return "0." + f"{float(x):.03f}".split(".")[1]

    def highlighting(x):
        return f"\\textbf{{{float_fmt(x)}}}"

    if skip_header:
        header = table[0]
        table = table[1:]
    for table_row in table:
        for i, cell in enumerate(table_row):
            try:
                float(cell)
                table_row[i] = float_fmt(cell)
            except ValueError:
                pass
    if axis == "row":
        for i, row in enumerate(table):
            row_values = [float(val) for val in row[1:]]
            if is_max:
                max_val = max(row_values)
                ids = [i + 1 for i, val in enumerate(row_values) if val == max_val]
            else:
                min_val = min(row_values)
                ids = [i + 1 for i, val in enumerate(row_values) if val == min_val]
            for id in ids:
                table[i][id] = highlighting(table[i][id])
    elif axis == "column":
        for i in range(1, len(table[0])):
            col_values = [float(row[i]) for row in table]
            if is_max:
                max_val = max(col_values)
                ids = [i for i, val in enumerate(col_values) if val == max_val]
            else:
                min_val = min(col_values)
                ids = [i for i, val in enumerate(col_values) if val == min_val]
            for id in ids:
                table[id][i] = highlighting(table[id][i])
    elif axis is None:
        table_values = [[float(val) for val in row[1:]] for row in table]
        if is_max:
            max_val = max([max(row) for row in table_values])
            ids = [
                (i, j + 1)
                for i, row in enumerate(table_values)
                for j, val in enumerate(row)
                if val == max_val
            ]
        else:
            min_val = min([min(row) for row in table_values])
            ids = [
                (i, j + 1)
                for i, row in enumerate(table_values)
                for j, val in enumerate(row)
                if val == min_val
            ]
        for i, j in ids:
            table[i][j] = highlighting(table[i][j])
    else:
        raise ValueError("Invalid axis, it must be either 'row', 'column', or None")
    if skip_header:
        table = [header] + table
    return table


def pairwise_compare(
    evaluator1: str | list,
    evaluator2: str | list,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Compare pairwise evaluators.

    Args:
        evaluator1: The responses from the first evaluator, either a list of responses or the path to the responses.
        evaluator2: The responses from the second evaluator, either a list of responses or the path to the responses.
        verbose: Whether to print verbose output.

    Returns:
        None
    """
    if isinstance(evaluator1, str):
        evaluator1_responses = read_json(evaluator1)
    else:
        evaluator1_responses = evaluator1
    if isinstance(evaluator2, str):
        evaluator2_responses = read_json(evaluator2)
    else:
        evaluator2_responses = evaluator2
    assert len(evaluator1_responses) == len(evaluator2_responses)
    evaluator1_winners = np.array(
        [response["winner"] for response in evaluator1_responses]
    )
    evaluator2_winners = np.array(
        [response["winner"] for response in evaluator2_responses]
    )
    acc = (evaluator1_winners == evaluator2_winners).mean().item()
    # agreement = cohen_kappa_score(evaluator1_winners, evaluator2_winners)
    agreement = krippendorff.alpha(
        [evaluator1_winners, evaluator2_winners], level_of_measurement="nominal"
    )
    if verbose:
        print(f"Agreement: {agreement}")
        print(f"Accuracy: {acc}")
    return acc, agreement


def pairwise_meta_eval(
    human: str | list,
    model: str | list,
    model_swap: str | list | None = None,
    verbose: bool = True,
) -> dict[float]:
    """
    Evaluate a pairwise evaluator.

    Args:
        human: The responses from the human evaluator, either a list of responses or the path to the responses.
        model: The responses from the model evaluator, either a list of responses or the path to the responses.
        model_swap: The responses from the model evaluator with swapped winners, either a list of responses or the path to the responses.
        verbose: Whether to print verbose output.

    Returns:
        dict[float]: The accuracy and agreement.
    """
    acc, agr = pairwise_compare(human, model, verbose=False)
    if model_swap is not None:
        swap_acc, swap_agr = pairwise_compare(human, model_swap, verbose=False)
        acc = (acc + swap_acc) / 2
        agr = (agr + swap_agr) / 2
        models_acc, models_agr = pairwise_compare(model, model_swap, verbose=False)
    result = {"acc": acc, "agr": agr}
    if model_swap is not None:
        result["models_acc"] = models_acc
        result["models_agr"] = models_agr
    if verbose:
        print(f"Accuracy: {acc}")
        print(f"Agreement: {agr}")
        if model_swap is not None:
            print(f"Model Accuracy: {models_acc}")
            print(f"Model Agreement: {models_agr}")
    return result


def meta_eval_main(args):
    parser = argparse.ArgumentParser("Run Pairwise Meta-Evaluation")
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument(
        "--evaluators_dir",
        type=str,
        default=None,
        help="Path to the evaluators yaml file",
    )
    parser.add_argument(
        "--methods_dir", type=str, default=None, help="Path to the methods yaml file"
    )
    parser.add_argument(
        "--models_dir", type=str, default=None, help="Path to the models yaml file"
    )
    parser.add_argument(
        "--aggregate_methods",
        action="store_true",
        help="Whether to aggregate the method results",
    )
    parser.add_argument(
        "--aggregate_models",
        action="store_true",
        help="Whether to aggregate the model results",
    )
    parser.add_argument(
        "--aggregate_datasets",
        action="store_true",
        help="Whether to aggregate the dataset results",
    )
    parser.add_argument(
        "--sorted", action="store_true", help="Whether to sort the results"
    )
    parser.add_argument(
        "--use_cache", action="store_true", help="Whether to use the cache"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cal_avg", action="store_true")
    parser.add_argument("--table_dir", type=str, default=None)
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument(
        "--aggregation_mode",
        type=str,
        default="mean",
        choices=["mean", "max", "min", "delta"],
    )
    parser.add_argument("--base_method", type=str, default="base")
    args = parser.parse_args(args)
    if args.debug:
        bugs = []

    if args.evaluators_dir is None and (
        args.methods_dir is None or args.models_dir is None
    ):
        raise ValueError(
            "Either evaluators_dir or methods_dir and models_dir must be provided"
        )

    if args.evaluators_dir is not None and (
        args.methods_dir is not None or args.models_dir is not None
    ):
        raise ValueError(
            "Only one of evaluators_dir or methods_dir and models_dir must be provided"
        )

    datasets = args.datasets
    if args.evaluators_dir is not None:
        with open(args.evaluators_dir) as f:
            evaluators = yaml.safe_load(f)
        print(evaluators)
        evaluator_names = [evaluator["name"] for evaluator in evaluators]

        human_model_accs = {evaluator: dict() for evaluator in evaluator_names}
        human_model_agrs = {evaluator: dict() for evaluator in evaluator_names}
        model_accs = {evaluator: defaultdict(str) for evaluator in evaluator_names}
        model_agrs = {evaluator: defaultdict(str) for evaluator in evaluator_names}

        for evaluator in evaluators:
            for dataset in datasets:
                print(evaluator["name"], dataset)
                human_dir = get_dataset_path(dataset)
                fdir = f"results/{dataset}.{evaluator['fdir']}.jsonl"
                if evaluator["bidirectional"]:
                    swap_fdir = f"results/{dataset}.{evaluator['fdir']}_swap.jsonl"
                else:
                    swap_fdir = None
                results = pairwise_meta_eval(human_dir, fdir, swap_fdir)
                human_model_accs[evaluator["name"]][dataset] = results["acc"]
                human_model_agrs[evaluator["name"]][dataset] = results["agr"]
                if swap_fdir is not None:
                    model_accs[evaluator["name"]][dataset] = results["models_acc"]
                    model_agrs[evaluator["name"]][dataset] = results["models_agr"]

        def print_table(results):
            table = [["Model"] + datasets]
            for evaluator in evaluator_names:
                row = [evaluator]
                for dataset in datasets:
                    row += [results[evaluator][dataset]]
                table.append(row)
            # table = highlight(table, axis="column")
            print(tabulate(table, headers="firstrow", floatfmt=".3f"))

        print("Human Evaluator Accuracies")
        print_table(human_model_accs)

        print("Human Evaluator Agreements")
        print_table(human_model_agrs)

        print("Evaluator Self Accuracies")
        print_table(model_accs)

        print("Evaluator Self Agreements")
        print_table(model_agrs)
    else:
        with open(args.methods_dir) as f:
            methods = yaml.safe_load(f)
        method_names = [method["name"] for method in methods]
        with open(args.models_dir) as f:
            models = yaml.safe_load(f)
        model_names = [model["name"] for model in models]

        human_model_accs = {
            dataset: {method: dict() for method in method_names} for dataset in datasets
        }
        human_model_agrs = {
            dataset: {method: dict() for method in method_names} for dataset in datasets
        }
        model_accs = {
            dataset: {method: defaultdict(str) for method in method_names}
            for dataset in datasets
        }
        model_agrs = {
            dataset: {method: defaultdict(str) for method in method_names}
            for dataset in datasets
        }

        for dataset in datasets:
            human_dir = get_dataset_path(dataset)
            for method in methods:
                for model in models:
                    print(dataset, method["name"], model["name"])
                    if "fdir_ext" in method:
                        fdir = f"results/{dataset}.{model['fdir']}.{method['fdir']}.{method['fdir_ext']}.jsonl"
                    else:
                        fdir = (
                            f"results/{dataset}.{model['fdir']}.{method['fdir']}.jsonl"
                        )
                    if method["bidirectional"]:
                        if "fdir_ext" in method:
                            swap_fdir = f"results/{dataset}.{model['fdir']}.{method['fdir']}_swap.{method['fdir_ext']}.jsonl"
                        else:
                            swap_fdir = f"results/{dataset}.{model['fdir']}.{method['fdir']}_swap.jsonl"
                    else:
                        swap_fdir = None
                    file_miss = False
                    if args.debug:
                        if not os.path.exists(fdir):
                            print(f"Warning: {fdir} does not exist")
                        if swap_fdir is not None and not os.path.exists(swap_fdir):
                            print(f"Warning: {swap_fdir} does not exist")
                        if not os.path.exists(fdir) or (
                            swap_fdir is not None and not os.path.exists(swap_fdir)
                        ):
                            results = {
                                "acc": 0,
                                "agr": 0,
                                "models_acc": 0,
                                "models_agr": 0,
                            }
                            file_miss = True
                            bugs.append(
                                (
                                    "file_miss",
                                    dataset,
                                    method["name"],
                                    model["name"],
                                    fdir,
                                    swap_fdir,
                                )
                            )
                    if not file_miss:
                        if args.debug:
                            try:
                                results = pairwise_meta_eval(
                                    human_dir, fdir, swap_fdir, verbose=args.verbose
                                )
                            except Exception as e:
                                bugs.append(
                                    (
                                        "exception",
                                        dataset,
                                        method["name"],
                                        model["name"],
                                        str(e),
                                    )
                                )
                                print(e)
                                results = {
                                    "acc": 0,
                                    "agr": 0,
                                    "models_acc": 0,
                                    "models_agr": 0,
                                }
                        else:
                            result_dir = os.path.join(
                                "results/meta_eval_cache",
                                f"{dataset}.{model['name']}.{method['name']}.json",
                            )
                            if args.use_cache and os.path.exists(result_dir):
                                with open(result_dir) as f:
                                    results = json.load(f)
                            else:
                                results = pairwise_meta_eval(
                                    human_dir, fdir, swap_fdir, verbose=args.verbose
                                )
                                with open(result_dir, "w") as f:
                                    json.dump(results, f, indent=2)
                    human_model_accs[dataset][method["name"]][model["name"]] = results[
                        "acc"
                    ]
                    human_model_agrs[dataset][method["name"]][model["name"]] = results[
                        "agr"
                    ]
                    if swap_fdir is not None:
                        model_accs[dataset][method["name"]][model["name"]] = results[
                            "models_acc"
                        ]
                        model_agrs[dataset][method["name"]][model["name"]] = results[
                            "models_agr"
                        ]

        if args.aggregate_methods and args.aggregate_models and args.aggregate_datasets:
            raise ValueError(
                "Cannot aggregate methods, models, and datasets at the same time"
            )
        elif args.aggregate_methods and args.aggregate_models:

            def print_table(results, table_dir=None):
                table = [["Dataset", "Result"]]
                if args.cal_avg:
                    avg_result = 0
                for dataset in datasets:
                    row = [dataset]
                    _results = []
                    for method in method_names:
                        for model in model_names:
                            _results.append(results[dataset][method][model])
                    if args.aggregation_mode == "mean":
                        result = np.mean(_results).item()
                    elif args.aggregation_mode == "max":
                        result = np.max(_results).item()
                    elif args.aggregation_mode == "min":
                        result = np.min(_results).item()
                    row += [result]
                    table.append(row)
                    if args.cal_avg:
                        avg_result += result
                if args.sorted:
                    table = [table[0]] + sorted(
                        table[1:], key=lambda x: x[1], reverse=True
                    )
                if args.cal_avg:
                    avg_result /= len(datasets)
                    table.append(["Avg", avg_result])
                latex_table = latex_highlight(table, axis="column")
                if table_dir is not None:
                    with open(table_dir, "w") as f:
                        f.write(
                            tabulate(
                                latex_table,
                                headers="firstrow",
                                tablefmt="latex_raw",
                                disable_numparse=True,
                            )
                        )
                else:
                    print(
                        tabulate(
                            latex_table,
                            headers="firstrow",
                            tablefmt="latex_raw",
                            disable_numparse=True,
                        )
                    )
                table = highlight(table, axis="column")
                print(tabulate(table, headers="firstrow", floatfmt=".3f"))

        elif args.aggregate_methods and args.aggregate_datasets:

            def print_table(results, table_dir=None):
                table = [["Model", "Result"]]
                if args.cal_avg:
                    avg_result = 0
                for model in model_names:
                    row = [model]
                    _results = []
                    for dataset in datasets:
                        for method in method_names:
                            _results.append(results[dataset][method][model])
                    if args.aggregation_mode == "mean":
                        result = np.mean(_results).item()
                    elif args.aggregation_mode == "max":
                        result = np.max(_results).item()
                    elif args.aggregation_mode == "min":
                        result = np.min(_results).item()
                    row += [result]
                    table.append(row)
                    if args.cal_avg:
                        avg_result += result
                if args.sorted:
                    table = [table[0]] + sorted(
                        table[1:], key=lambda x: x[1], reverse=True
                    )
                if args.cal_avg:
                    avg_result /= len(model_names)
                    table.append(["Avg", avg_result])
                latex_table = latex_highlight(table, axis="column")
                if table_dir is not None:
                    with open(table_dir, "w") as f:
                        f.write(
                            tabulate(
                                latex_table,
                                headers="firstrow",
                                tablefmt="latex_raw",
                                disable_numparse=True,
                            )
                        )
                else:
                    print(
                        tabulate(
                            latex_table,
                            headers="firstrow",
                            tablefmt="latex_raw",
                            disable_numparse=True,
                        )
                    )
                table = highlight(table, axis="column")
                print(tabulate(table, headers="firstrow", floatfmt=".3f"))

        elif args.aggregate_models and args.aggregate_datasets:

            def print_table(results, table_dir=None):
                table = [["Method", "Result"]]
                if args.cal_avg:
                    avg_result = 0
                for method in method_names:
                    row = [method]
                    _results = []
                    for dataset in datasets:
                        for model in model_names:
                            _results.append(results[dataset][method][model])
                    if args.aggregation_mode == "mean":
                        result = np.mean(_results).item()
                    elif args.aggregation_mode == "max":
                        result = np.max(_results).item()
                    elif args.aggregation_mode == "min":
                        result = np.min(_results).item()
                    row += [result]
                    table.append(row)
                    if args.cal_avg:
                        avg_result += result
                if args.sorted:
                    table = [table[0]] + sorted(
                        table[1:], key=lambda x: x[1], reverse=True
                    )
                if args.cal_avg:
                    avg_result /= len(method_names)
                    table.append(["Avg", avg_result])
                latex_table = latex_highlight(table, axis="column")
                if table_dir is not None:
                    with open(table_dir, "w") as f:
                        f.write(
                            tabulate(
                                latex_table,
                                headers="firstrow",
                                tablefmt="latex_raw",
                                disable_numparse=True,
                            )
                        )
                else:
                    print(
                        tabulate(
                            latex_table,
                            headers="firstrow",
                            tablefmt="latex_raw",
                            disable_numparse=True,
                        )
                    )
                table = highlight(table, axis="column")
                print(tabulate(table, headers="firstrow", floatfmt=".3f"))

        elif args.aggregate_methods:

            def print_table(results, table_dir=None):
                table = [["Model"] + datasets]
                if args.cal_avg:
                    table[0].append("Avg")
                for model in model_names:
                    row = [model]
                    if args.cal_avg:
                        avg_result = 0
                    for dataset in datasets:
                        _results = []
                        for method in method_names:
                            _results.append(results[dataset][method][model])
                        if args.aggregation_mode == "mean":
                            result = np.mean(_results).item()
                        elif args.aggregation_mode == "max":
                            result = np.max(_results).item()
                        elif args.aggregation_mode == "min":
                            result = np.min(_results).item()
                        elif args.aggregation_mode == "delta":
                            base_result = results[dataset][args.base_method][model]
                            _results = [
                                results[dataset][method][model]
                                for method in method_names
                                if method != args.base_method
                            ]
                            best_result = max(_results)
                            result = best_result - base_result
                        if args.cal_avg:
                            avg_result += result
                        row += [result]
                    if args.cal_avg:
                        avg_result /= len(datasets)
                        row.append(avg_result)
                    table.append(row)
                if args.sorted and args.cal_avg:
                    table = [table[0]] + sorted(
                        table[1:], key=lambda x: x[-1], reverse=True
                    )
                latex_table = latex_highlight(table, axis="column")
                if table_dir is not None:
                    with open(table_dir, "w") as f:
                        f.write(
                            tabulate(
                                latex_table,
                                headers="firstrow",
                                tablefmt="latex_raw",
                                disable_numparse=True,
                            )
                        )
                else:
                    print(
                        tabulate(
                            latex_table,
                            headers="firstrow",
                            tablefmt="latex_raw",
                            disable_numparse=True,
                        )
                    )
                table = highlight(table, axis="column")
                print(tabulate(table, headers="firstrow", floatfmt=".3f"))

        elif args.aggregate_models:

            def print_table(results, table_dir=None):
                table = [["Method"] + datasets]
                if args.cal_avg:
                    table[0].append("Avg")
                for method in method_names:
                    row = [method]
                    if args.cal_avg:
                        avg_result = 0
                    for dataset in datasets:
                        _results = []
                        for model in model_names:
                            _results.append(results[dataset][method][model])
                        if args.aggregation_mode == "mean":
                            result = np.mean(_results).item()
                        elif args.aggregation_mode == "max":
                            result = np.max(_results).item()
                        elif args.aggregation_mode == "min":
                            result = np.min(_results).item()
                        if args.cal_avg:
                            avg_result += result
                        row += [result]
                    if args.cal_avg:
                        avg_result /= len(datasets)
                        row.append(avg_result)
                    table.append(row)
                if args.sorted and args.cal_avg:
                    table = [table[0]] + sorted(
                        table[1:], key=lambda x: x[-1], reverse=True
                    )
                latex_table = latex_highlight(table, axis="column")
                if table_dir is not None:
                    with open(table_dir, "w") as f:
                        f.write(
                            tabulate(
                                latex_table,
                                headers="firstrow",
                                tablefmt="latex_raw",
                                disable_numparse=True,
                            )
                        )
                else:
                    print(
                        tabulate(
                            latex_table,
                            headers="firstrow",
                            tablefmt="latex_raw",
                            disable_numparse=True,
                        )
                    )
                table = highlight(table, axis="column")
                print(tabulate(table, headers="firstrow", floatfmt=".3f"))

        elif args.aggregate_datasets:
            if args.transpose:

                def print_table(results, table_dir=None):
                    table = [["Model/Method"] + method_names]
                    if args.cal_avg:
                        table[0].append("Avg")
                    for model in model_names:
                        row = [model]
                        if args.cal_avg:
                            avg_result = 0
                        for method in method_names:
                            _results = []
                            for dataset in datasets:
                                _results.append(results[dataset][method][model])
                            if args.aggregation_mode == "mean":
                                result = np.mean(_results).item()
                            elif args.aggregation_mode == "max":
                                result = np.max(_results).item()
                            elif args.aggregation_mode == "min":
                                result = np.min(_results).item()
                            if args.cal_avg:
                                avg_result += result
                            row += [result]
                        if args.cal_avg:
                            avg_result /= len(method_names)
                            row.append(avg_result)
                        table.append(row)
                    if args.sorted and args.cal_avg:
                        table = [table[0]] + sorted(
                            table[1:], key=lambda x: x[-1], reverse=True
                        )
                    latex_table = latex_highlight(table)
                    if table_dir is not None:
                        with open(table_dir, "w") as f:
                            f.write(
                                tabulate(
                                    latex_table,
                                    headers="firstrow",
                                    tablefmt="latex_raw",
                                    disable_numparse=True,
                                )
                            )
                    else:
                        print(
                            tabulate(
                                latex_table,
                                headers="firstrow",
                                tablefmt="latex_raw",
                                disable_numparse=True,
                            )
                        )
                    table = highlight(table)
                    print(tabulate(table, headers="firstrow", floatfmt=".3f"))

            else:

                def print_table(results, table_dir=None):
                    table = [["Method/Model"] + model_names]
                    if args.cal_avg:
                        table[0].append("Avg")
                    for method in method_names:
                        row = [method]
                        if args.cal_avg:
                            avg_result = 0
                        for model in model_names:
                            _results = []
                            for dataset in datasets:
                                _results.append(results[dataset][method][model])
                            if args.aggregation_mode == "mean":
                                result = np.mean(_results).item()
                            elif args.aggregation_mode == "max":
                                result = np.max(_results).item()
                            elif args.aggregation_mode == "min":
                                result = np.min(_results).item()
                            if args.cal_avg:
                                avg_result += result
                            row += [result]
                        if args.cal_avg:
                            avg_result /= len(model_names)
                            row.append(avg_result)
                        table.append(row)
                    if args.sorted and args.cal_avg:
                        table = [table[0]] + sorted(
                            table[1:], key=lambda x: x[-1], reverse=True
                        )
                    latex_table = latex_highlight(table)
                    if table_dir is not None:
                        with open(table_dir, "w") as f:
                            f.write(
                                tabulate(
                                    latex_table,
                                    headers="firstrow",
                                    tablefmt="latex_raw",
                                    disable_numparse=True,
                                )
                            )
                    else:
                        print(
                            tabulate(
                                latex_table,
                                headers="firstrow",
                                tablefmt="latex_raw",
                                disable_numparse=True,
                            )
                        )
                    table = highlight(table)
                    print(tabulate(table, headers="firstrow", floatfmt=".3f"))

        else:

            def print_table(results):
                table = [["Dataset", "Method", "Model", "Result"]]
                for dataset in datasets:
                    for method in method_names:
                        for model in model_names:
                            row = [dataset, method, model]
                            row += [results[dataset][method][model]]
                            table.append(row)
                if args.sorted:
                    table = [table[0]] + sorted(
                        table[1:], key=lambda x: x[3], reverse=True
                    )
                table = highlight(table)
                print(tabulate(table, headers="firstrow", floatfmt=".3f"))

        print("Human Evaluator Accuracies")
        if args.table_dir is not None:
            table_dir = args.table_dir + "_human_model_accs.tex"
        else:
            table_dir = None
        print_table(human_model_accs, table_dir)
        print("-" * 80)

        print("Evaluator Self Agreements")
        if args.table_dir is not None:
            table_dir = args.table_dir + "_model_agrs.tex"
        else:
            table_dir = None
        print_table(model_agrs, table_dir)
        print("-" * 80)

    if args.debug:
        for bug in bugs:
            print(bug)
            print("-" * 80)

    print("Number of models: ", len(model_names))


if __name__ == "__main__":
    meta_eval_main()
