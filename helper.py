from ReIFE.methods import list_methods, method_info, list_parsers, parser_info
from ReIFE.models import list_models, model_info
import argparse


def main():
    parser = argparse.ArgumentParser("helper")
    parser.add_argument("--list_methods", action="store_true")
    parser.add_argument("--method_info", type=str)
    parser.add_argument("--list_parsers", action="store_true")
    parser.add_argument("--parser_info", type=str)
    parser.add_argument("--list_models", action="store_true")
    parser.add_argument("--model_info", type=str)

    args = parser.parse_args()
    if args.list_methods:
        list_methods()
    if args.method_info:
        method_info(args.method_info)
    if args.list_parsers:
        list_parsers()
    if args.parser_info:
        parser_info(args.parser_info)
    if args.list_models:
        list_models()
    if args.model_info:
        model_info(args.model_info)


if __name__ == "__main__":
    main()
