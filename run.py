from utils import get_dataset_verbalizers
from DELVEPO import DELVEPO
from typing import List
import argparse
import random
import json

def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--population_size", type=int, default=10, help="The population size of prompts"
    )
    arg_parser.add_argument(
        "--epoch_size", type=int, default=10, help="The epoch size"
    ) 
    arg_parser.add_argument(
        "--sample_num", type=int, default=100, help="The number of samples used to choose the optimized sequences"
    )
    arg_parser.add_argument(
        "--factor_size", type=int, default=10, help="The size of factor set"
    ) 
    arg_parser.add_argument(
        "--factor_path", type=str, 
        default="data/cls/sst-5/factors.json",
        help="The path to initial factor types and factor sets"
    ) 
    arg_parser.add_argument(
        "--number_of_pairs", type=int, default=10, help="The number of memory pairs"
    ) 
    arg_parser.add_argument(
        "--number_of_set", type=int, default=10, help="The number of memory set"
    ) 
    arg_parser.add_argument(
        "--parents_size", type=int, default=10, help="The parent size"
    ) 
    arg_parser.add_argument(
        "--task", type=str, default="cls", help="task in [cls, nlg, nlu, reason]"
    )
    arg_parser.add_argument(
        "--dataset", type=str, default="sst-5", help="dataset in [sst-5, casual_judgement, trec, StrategyQA, CommonSenseQA, SAMSum, SQuAD, ...]" 
    )
    arg_parser.add_argument("--llm_type", type=str, default="deepseek-r1-8b")
    arg_parser.add_argument("--seed", type=int, default=5, help="random seed")
    arg_parser.add_argument("--data_path", type=str, default="data/cls/sst-5/dev.txt") 
    arg_parser.add_argument("--output_path", type=str, default="outputs/")
    arg_info = arg_parser.parse_args(args=in_args)
    arg_info = load_factors(arg_info, arg_info.factor_size)
    return arg_info

def parse_factor_type(value: str) -> List[List[str]]:
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {value}")

def load_factors(args, factor_size):
    """Loads factor_type and factor_set from factor_path"""
    random.seed(args.seed)
    factor_path = args.factor_path
    with open(factor_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    factor_types = data.get("FactorType", [])
    factor_sets = data.get("FactorSet", [])
    if len(factor_types) != len(factor_sets):
        raise ValueError("The number of FactorType elements must match the number of FactorSet lists.")
    new_factor_sets = []
    for factor_list in factor_sets:
        if len(factor_list) <= factor_size:
            sampled_list = factor_list.copy()
        else:
            sampled_list = random.sample(factor_list, factor_size)
        new_factor_sets.append(sampled_list)
    print(f"factor_types: ({len(factor_types)}): {factor_types}")
    print(f"factor_sets: ({len(factor_sets)}): {factor_sets}")
    setattr(args, "factor_type", factor_types)
    setattr(args, "factor_set", new_factor_sets)
    return args

if __name__ == "__main__":
    in_argv = parse_args()
    if in_argv.task == "cls":
        verbalizers = get_dataset_verbalizers(in_argv.dataset)
    else:
        verbalizers = None
    task = DELVEPO(in_argv, verbalizers=verbalizers)
    best_valids, best_prompt = task.run_self_evolution()
    print(f"best_prompt = {best_prompt[-1]}, best_metrics = {best_valids[-1]}")
