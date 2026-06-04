"""Benchmark ZipServ decompression from .pt files."""

import argparse
import os
import time

import torch

import zipserv


def main(compress_dir: str, layer_types: list[str] | None, layer_idxs: list[int] | None, device: str, warmup: int, iters: int, profile: bool):
    files = sorted(f for f in os.listdir(compress_dir) if f.endswith(".pt"))
    if not files:
        print(f"No .pt files found in {compress_dir}")
        return

    # Apply layer index filter
    if layer_idxs is not None:
        matched = []
        for f in files:
            stem = f.removesuffix(".pt")
            if any(f"_layers_{idx}_" in stem for idx in layer_idxs):
                matched.append(f)
        if not matched:
            print(f"No layers matched indices: {layer_idxs}")
            print(f"Available layer stems: {[f.removesuffix('.pt') for f in files]}")
            return
        files = matched
        print(f"Filtered to {len(files)} layer(s) with index: {layer_idxs}")

    # Apply layer type filter
    if layer_types is not None:
        matched = []
        for f in files:
            stem = f.removesuffix(".pt")
            if any(t in stem for t in layer_types):
                matched.append(f)
        if not matched:
            print(f"No layers matched types: {layer_types}")
            print(f"Available layer stems: {[f.removesuffix('.pt') for f in files]}")
            return
        files = matched
        print(f"Filtered to {len(files)} layer(s) with type: {layer_types}")

    print(f"Found {len(files)} compressed file(s) in {compress_dir}")
    print(f"Device: {device}\n")

    # Load all compressed tensors once + pre-allocate output buffers
    compressed_tensors = []
    total_elements = 0

    for fn in files:
        path = os.path.join(compress_dir, fn)
        state = torch.load(path, map_location="cpu", weights_only=True)
        c = zipserv.CompressedTensor.from_state_dict(state)

        # Pre-stage all metadata on the target device
        c.to(device, non_blocking=True)

        # Pre-allocate output buffer so decompress can reuse it
        M, K = c.orig_shape
        buf = torch.empty((M, K), dtype=torch.bfloat16, device=device)

        compressed_tensors.append((fn, c, buf))
        total_elements += M * K

    print(f"Total elements to decompress: {total_elements:,}")
    print(f"Total size: {total_elements * 2 / 1024**3:.2f} GB (BF16)\n")

    use_cuda_event = device.startswith("cuda")

    # Warmup
    for _ in range(warmup):
        for _, c, buf in compressed_tensors:
            if profile:
                c.decompress(device, output=buf, profile=profile)
            else:
                c.decompress(device, output=buf)
    if use_cuda_event:
        torch.cuda.synchronize()

    # Benchmark
    if use_cuda_event:
        timings = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _, c, buf in compressed_tensors:
                if profile:
                    c.decompress(device, output=buf, profile=profile)
                else:
                    c.decompress(device, output=buf)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))  # ms
        elapsed_ms_per_iter = sum(timings) / len(timings)
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            for _, c, buf in compressed_tensors:
                if profile:
                    c.decompress(device, output=buf, profile=profile)
                else:
                    c.decompress(device, output=buf)
        t1 = time.perf_counter()
        elapsed_ms_per_iter = ((t1 - t0) / iters) * 1000

    elapsed_sec = elapsed_ms_per_iter / 1000.0

    print("=" * 60)
    print(f"Warmup iterations : {warmup}")
    print(f"Benchmark iterations: {iters}")
    if use_cuda_event:
        print(f"Time per iteration: {elapsed_ms_per_iter:.2f} ms (CUDA event)")
    else:
        print(f"Time per iteration: {elapsed_ms_per_iter:.2f} ms (wall clock)")
    print(f"Throughput        : {total_elements * 2 / elapsed_sec / 1024**3:.2f} GB/s")
    print(f"Elements/sec      : {total_elements / elapsed_sec:.2e}")
    print("=" * 60)

    if profile and use_cuda_event:
        # Run one more iteration and collect profiling data
        torch.cuda.synchronize()
        all_prof = []
        for _, c, buf in compressed_tensors:
            _, prof = c.decompress(device, output=buf, profile=True)
            all_prof.append(prof)
        torch.cuda.synchronize()

        # Query actual GPU SM clock rate (reported in kHz)
        props = torch.cuda.get_device_properties(device)
        clock_rate_khz = props.clock_rate
        cycles_per_ms = clock_rate_khz
        num_sms = props.multi_processor_count

        phase_names = ["HBM→SMEM load", "Offset resolve", "SMEM decompression", "Global write-back"]
        for fn, prof in zip(files, all_prof):
            num_ctas = prof.shape[0]
            prof_cpu = prof.cpu().numpy()
            deltas = prof_cpu[:, 1:] - prof_cpu[:, :-1]
            avg_cycles = deltas.mean(axis=0)
            print(f"\nProfile for {fn}:")
            print(f"  CTAs: {num_ctas}, SMs: {num_sms}, CTAs/SM: {num_ctas / num_sms:.1f}")
            for name, cycles in zip(phase_names, avg_cycles):
                ms = cycles / cycles_per_ms
                pct = (cycles / avg_cycles.sum()) * 100 if avg_cycles.sum() > 0 else 0
                print(f"  {name:20s}: {cycles:12.0f} cycles  ({ms:8.3f} ms, {pct:5.1f}%)")
    elif profile and not use_cuda_event:
        print("Profiling is only supported on CUDA devices.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark ZipServ decompression speed"
    )
    parser.add_argument(
        "--in-dir",
        type=str,
        required=True,
        help="Directory containing compressed .pt files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to decompress to (default: cuda)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--layer-type",
        type=str,
        nargs="+",
        default=None,
        dest="layer_types",
        help="Substrings to match in layer names (e.g., 'q_proj', 'mlp'). "
             "If omitted, all types are included.",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        nargs="+",
        default=None,
        dest="layer_idxs",
        help="Decoder layer indices to select (e.g., 0 1). "
             "If omitted, all indices are included.",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=20,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-block kernel phase profiling (CUDA only)",
    )
    args = parser.parse_args()
    main(args.in_dir, args.layer_types, args.layer_idxs, args.device, args.warmup, args.iter, args.profile)
