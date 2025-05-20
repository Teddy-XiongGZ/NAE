import json
import argparse
from config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exp_str", type=str, help="special string to identify an experiment")
    parser.add_argument("--var_penalty", type=float, default=None)
    parser.add_argument("--C", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    config_params = json.load(open(f"./configs/{args.data}_{args.model}_optim.json"))
    
    if args.C is not None:
        config_params["C"] = args.C
    if args.k is not None:
        config_params["k"] = args.k
    if args.var_penalty is not None:
        config_params["var_penalty"] = args.var_penalty

    config = Config(
        **config_params,
        data = args.data,
        model = args.model,
        device = args.device,
        exp_str = args.exp_str,
        fold = args.fold,
        seed = args.seed,
    )
        
    config.train()