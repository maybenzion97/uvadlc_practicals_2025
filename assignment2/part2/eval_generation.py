"""
Evaluation script for Q2.8(b): probing the model's accumulated knowledge
under different decoding settings.

Example usage:
    python eval_generation.py \
        --model_weights_folder ./logs/gpt-mini/version_0/checkpoints \
        --output_file q2_8b_results.json
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Union

import torch
import pytorch_lightning as pl

from cfg import get_config
from gpt import GPT
from dataset import TextDataset, CharTokenizer
from generate import generate, GPTLightningModule


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pick_device() -> str:
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(checkpoint_dir: str) -> Dict[str, Any]:
    """Load the most recent checkpoint from a directory."""
    files = sorted(os.listdir(checkpoint_dir))
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in '{checkpoint_dir}'")
    ckpt_path = os.path.join(checkpoint_dir, files[-1])
    print(f"Loading checkpoint: {ckpt_path}")
    # We keep weights_only=False for maximum compatibility with Lightning checkpoints.
    state = torch.load(ckpt_path)
    if state['hyper_parameters'].get('compile', False) and 'state_dict' in state:
        cleaned_state_dict = {}
        for key, value in state['state_dict'].items():
            new_key = key.replace('model._orig_mod.', 'model.')
            cleaned_state_dict[new_key] = value
        state['state_dict'] = cleaned_state_dict

    return state


def reconstruct_cfg_from_state(state: Dict[str, Any]) -> argparse.Namespace:
    """Merge GPT default config with saved hyperparameters."""
    base_cfg = GPT.get_default_config()
    saved_hparams = argparse.Namespace(**state["hyper_parameters"])

    merged = {**vars(base_cfg), **vars(saved_hparams)}
    return argparse.Namespace(**merged)


def load_tokenizer_and_dataset(args: argparse.Namespace) -> Tuple[CharTokenizer, TextDataset]:
    """Create tokenizer and dataset in the same way as during training."""
    if args.pretrained_tokenizer:
        import tiktoken
        tok = tiktoken.get_encoding("gpt2")
        args.vocab_size = tok.max_token_value
    else:
        tok = CharTokenizer(args.txt_file)
        args.vocab_size = tok.vocab_size

    ds = TextDataset(args, args.txt_file, args.block_size, tok)
    return tok, ds


def coerce_generate_output(outputs: Union[str, List[str], Tuple[str, ...]]) -> str:
    """
    Make sure we get a single string back from `generate`, regardless of
    whether it returns a string or a sequence of strings.
    """
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        return str(outputs[0])
    raise RuntimeError(f"Unexpected output type from generate(): {type(outputs)}")


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "Once upon a time",
    "The king said to",
    "In the dark forest",
    "And they lived happily",
    "The old woman gave",
]

DEFAULT_CONFIGS = [
    # Greedy baseline
    {"name": "Greedy",                "do_sample": False, "top_k": None, "top_p": None, "temperature": 1.0},
    # Top-p / nucleus
    {"name": "Top-p (T=0.5, p=0.6)",  "do_sample": True,  "top_k": None, "top_p": 0.6,  "temperature": 0.5},
    {"name": "Top-p (T=0.8, p=0.6)",  "do_sample": True,  "top_k": None, "top_p": 0.6,  "temperature": 0.8},
    {"name": "Top-p (T=1.0, p=0.6)",  "do_sample": True,  "top_k": None, "top_p": 0.6,  "temperature": 1.0},
    {"name": "Top-p (T=1.0, p=0.9)",  "do_sample": True,  "top_k": None, "top_p": 0.9,  "temperature": 1.0},
    {"name": "Top-p (T=1.5, p=0.9)",  "do_sample": True,  "top_k": None, "top_p": 0.9,  "temperature": 1.5},
    # Top-k
    {"name": "Top-k (T=1.0, k=5)",    "do_sample": True,  "top_k": 5,    "top_p": None, "temperature": 1.0},
    {"name": "Top-k (T=1.0, k=20)",   "do_sample": True,  "top_k": 20,   "top_p": None, "temperature": 1.0},
    {"name": "Top-k (T=1.0, k=50)",   "do_sample": True,  "top_k": 50,   "top_p": None, "temperature": 1.0},
]


def parse_args() -> argparse.Namespace:
    base_args = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_weights_folder", type=str,
                        default="./logs/gpt-mini/version_0/checkpoints")
    parser.add_argument("--output_file", type=str,
                        default="generation_results.json")
    parser.add_argument("--num_generated_tokens", type=int, default=100)
    parser.add_argument("--pretrained_tokenizer", action="store_true")
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Optional JSON with 'prompts' and 'configurations' entries.",
    )
    extra = parser.parse_args()

    # Merge extra args into base config (overrides defaults from cfg.py)
    for k, v in vars(extra).items():
        setattr(base_args, k, v)
    return base_args


def load_prompts_and_configs(args: argparse.Namespace):
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, "r") as f:
            custom = json.load(f)
        prompts = custom.get("prompts") or DEFAULT_PROMPTS
        configs = custom.get("configurations") or DEFAULT_CONFIGS
    else:
        prompts = DEFAULT_PROMPTS
        configs = DEFAULT_CONFIGS
    return prompts, configs


def run_evaluation() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)

    print("Loading model checkpoint...")
    state = load_checkpoint(args.model_weights_folder)
    cfg = reconstruct_cfg_from_state(state)

    print("Rebuilding GPT model and tokenizer...")
    gpt_model = GPT(cfg)
    tokenizer, dataset = load_tokenizer_and_dataset(args)

    lightning_model = GPTLightningModule(cfg, gpt_model, dataset)
    lightning_model.load_state_dict(state["state_dict"])

    device = pick_device()
    lightning_model.to(device)
    lightning_model.eval()
    print(f"Running on device: {device}")

    prompts, configs = load_prompts_and_configs(args)

    results: Dict[str, Any] = {
        "metadata": {
            "model_weights_folder": args.model_weights_folder,
            "num_generated_tokens": args.num_generated_tokens,
            "seed": args.seed,
            "device": device,
        },
        "prompts": prompts,
        "configurations": configs,
        "generations": [],
    }

    print("=" * 100)
    print("GENERATION EVALUATION - Accumulated Knowledge")
    print("=" * 100)

    total_runs = len(prompts) * len(configs)
    run_idx = 0

    for prompt in prompts:
        print(f"\n{'=' * 100}")
        print(f'PROMPT: "{prompt}"')
        print("=" * 100)

        for cfg_dec in configs:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] {cfg_dec['name']}")

            try:
                outputs = generate(
                    model=lightning_model,
                    model_type=cfg.model_type,
                    prompt=prompt,
                    num_samples=1,
                    n_steps=args.num_generated_tokens,
                    do_sample=cfg_dec["do_sample"],
                    top_k=cfg_dec["top_k"],
                    top_p=cfg_dec["top_p"],
                    temperature=cfg_dec["temperature"],
                    device=device,
                    verbose=False,
                )

                full_text = coerce_generate_output(outputs)
                gen_only = full_text[len(prompt):]

                record = {
                    "prompt": prompt,
                    "config_name": cfg_dec["name"],
                    "do_sample": cfg_dec["do_sample"],
                    "top_k": cfg_dec["top_k"],
                    "top_p": cfg_dec["top_p"],
                    "temperature": cfg_dec["temperature"],
                    "full_output": full_text,
                    "generated_only": gen_only,
                    "error": None,
                }

                print("-" * 80)
                print(full_text[:200] + "..." if len(full_text) > 200 else full_text)

            except Exception as exc:
                print(f"ERROR: {exc}")
                record = {
                    "prompt": prompt,
                    "config_name": cfg_dec["name"],
                    "do_sample": cfg_dec["do_sample"],
                    "top_k": cfg_dec["top_k"],
                    "top_p": cfg_dec["top_p"],
                    "temperature": cfg_dec["temperature"],
                    "full_output": None,
                    "generated_only": None,
                    "error": str(exc),
                }

            results["generations"].append(record)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to: {args.output_file}")
    print(f"Total generations stored: {len(results['generations'])}")


if __name__ == "__main__":
    run_evaluation()