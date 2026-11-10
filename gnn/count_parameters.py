import argparse
import json
import os
import random

import numpy as np
import torch

from parsing import add_parse_args
from train_model import Trainer_KBQA


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
    }


def main():
    parser = argparse.ArgumentParser()
    add_parse_args(parser)
    args = parser.parse_args()

    args.use_cuda = torch.cuda.is_available()
    args.is_eval = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    trainer = Trainer_KBQA(args=vars(args), model_name=args.model_name, logger=None)
    counts = count_parameters(trainer.model)
    counts.update(
        {
            "model_name": args.model_name,
            "dataset": args.name,
            "seed": args.seed,
            "entity_dim": args.entity_dim,
            "kg_dim": args.kg_dim,
            "word_dim": args.word_dim,
            "lm": args.lm,
            "num_iter": getattr(args, "num_iter", None),
            "num_ins": getattr(args, "num_ins", None),
            "num_gnn": getattr(args, "num_gnn", None),
            "data_folder": args.data_folder,
            "checkpoint_dir": args.checkpoint_dir,
            "load_experiment": args.load_experiment,
        }
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
