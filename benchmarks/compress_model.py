"""ZipServ HuggingFace model compression benchmark."""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM

import zipserv


def get_model(model_name: str):
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )
    return model


def main(model_name: str, out: str):
    os.makedirs(out, exist_ok=True)
    print(f"Compress output directory: {os.path.abspath(out)}")

    model = get_model(model_name)

    n_saved = 0
    total_ratio = 0.0

    for name, param in model.named_parameters():
        if param.dim() != 2 or "weight" not in name:
            continue

        M, K = param.shape
        if M % 64 != 0 or K % 64 != 0:
            continue

        print(f"Compressing: {name}  shape={tuple(param.shape)}")
        mat_cpu = param.detach().cpu().bfloat16().contiguous()

        c = zipserv.compress(mat_cpu)
        total_ratio += c.ratio

        safe_name = name.replace(".", "_")
        out_path = os.path.join(out, f"{safe_name}.pt")
        torch.save(c.state_dict(), out_path)
        print(f"  Saved -> {out_path}  ratio={c.ratio:.2f}x")
        n_saved += 1

        if args.verify and torch.cuda.is_available():
            device = torch.device("cuda")
            c.to(device, non_blocking=True)
            original = mat_cpu.to(device, non_blocking=True)
            decompressed = c.decompress(device)
            match = (decompressed == original).sum().item()
            total = original.numel()
            max_err = (decompressed - original).abs().max().item()
            print(f"  Verify: {match}/{total} exact match  max_err={max_err}")

    if n_saved:
        print(f"\nTotal layers compressed: {n_saved}")
        print(f"Average compression ratio: {total_ratio / n_saved:.2f}x")
    else:
        print("No compressible layers found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ZipServ compression on a HuggingFace model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="benchmarks/compressed_output",
        help="Directory to store compressed output data",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=False,
        help="Verify round-trip correctness on CUDA when available",
    )
    args = parser.parse_args()
    main(args.model, args.out_dir)
