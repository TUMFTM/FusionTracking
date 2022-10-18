"""Helper script to prepare evaluation of a ros2 bag replay of the tracking module."""
import os
import sys
import argparse

REPO_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
sys.path.append(REPO_PATH)

from evaluation import Evaluation

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--results_save_path", type=str, default="evaluation/results")
    parser.add_argument("--save_file_name", type=str, default=None)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--show_plt", default=False, action="store_true")
    parser.add_argument("--print_terminal", default=False, action="store_true")
    args = parser.parse_args()

    # add non parse args
    args.load_path = None
    args.show_plt = False
    args.show_filtered = False

    eval = Evaluation(args)
    eval.prepare_data()
