import argparse
import os
from transformers import *
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("ensemble_path")
    parser.add_argument("--output_path", default="predictions.csv")

    args = parser.parse_args()

    models = []
    for model_path in os.listdir(args.ensemble_path):
        if not os.path.isdir(os.path.join(args.ensemble_path, model_path)):
            continue

        print(model_path)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.ensemble_path, model_path))
        model = torch.load(os.path.join(args.ensemble_path, model_path, "model.pt"))
