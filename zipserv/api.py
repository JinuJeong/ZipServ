"""Python-level helpers for ZipServ compression / decompression."""

import numpy as np
import torch

import zipserv


def _get_top_exponents(tensor: torch.Tensor, k: int = 7) -> torch.Tensor:
    """Return the top-*k* most frequent BF16 exponent values from a CPU tensor."""
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    arr = tensor.view(torch.uint16).numpy()
    exponents = (arr >> 7) & 0xFF
    unique, counts = np.unique(exponents, return_counts=True)
    order = np.argsort(counts)[::-1]
    top = unique[order[:k]]
    if len(top) < k:
        top = np.concatenate([top, np.zeros(k - len(top), dtype=np.int32)])
    return torch.from_numpy(top.astype(np.int32))


class CompressedTensor:
    """Container for a ZipServ triple-bitmap compressed BF16 matrix.

    Attributes
    ----------
    top_exponents : torch.Tensor
        The 7 most frequent exponent values (int32 CPU).
    sign_mantissa : torch.Tensor
        Sign+m bits for high-frequency elements (uint8 CPU).
    compressed_full : torch.Tensor
        Full BF16 values for non-high-frequency elements (bf16 CPU).
    bitmap1, bitmap2, bitmap3 : torch.Tensor
        Three 64-bit bitmaps (int64 CPU).
    tile_offsets : torch.Tensor
        Small tile offsets (int32 CPU, shape [num_tiles, 2]).
    tile_offsets_median : torch.Tensor
        Medium tile offsets (int32 CPU, shape [num_median_tiles, 2]).
    tile_offsets_global : torch.Tensor
        Large tile offsets (int32 CPU, shape [num_global_tiles+1, 2]).
    max_high_freq_count : int
        Maximum high-frequency element count per global tile.
    max_full_count : int
        Maximum full-value element count per global tile.
    num_global_tiles : int
        Number of global tiles.
    orig_shape : tuple[int, int]
        Original (M, K) shape before compression.
    """

    def __init__(
        self,
        top_exponents,
        sign_mantissa,
        compressed_full,
        bitmap1,
        bitmap2,
        bitmap3,
        tile_offsets,
        tile_offsets_median,
        tile_offsets_global,
        max_high_freq_count,
        max_full_count,
        num_global_tiles,
        orig_shape,
    ):
        self.top_exponents = top_exponents
        self.sign_mantissa = sign_mantissa
        self.compressed_full = compressed_full
        self.bitmap1 = bitmap1
        self.bitmap2 = bitmap2
        self.bitmap3 = bitmap3
        self.tile_offsets = tile_offsets
        self.tile_offsets_median = tile_offsets_median
        self.tile_offsets_global = tile_offsets_global
        self.max_high_freq_count = max_high_freq_count
        self.max_full_count = max_full_count
        self.num_global_tiles = num_global_tiles
        self.orig_shape = orig_shape

    @classmethod
    def _from_compress_tuple(cls, tup, orig_shape):
        """Build from the raw 12-element tuple returned by C++."""
        return cls(
            top_exponents=tup[0],
            sign_mantissa=tup[1],
            compressed_full=tup[2],
            bitmap1=tup[3],
            bitmap2=tup[4],
            bitmap3=tup[5],
            tile_offsets=tup[6],
            tile_offsets_median=tup[7],
            tile_offsets_global=tup[8],
            max_high_freq_count=tup[9],
            max_full_count=tup[10],
            num_global_tiles=tup[11],
            orig_shape=orig_shape,
        )

    @property
    def start_exp(self) -> int:
        """The starting exponent used for encoding (first top exponent)."""
        return int(self.top_exponents[0].item())

    @property
    def ratio(self) -> float:
        """Compression ratio (original bytes / compressed bytes)."""
        M, K = self.orig_shape
        orig_nbytes = M * K * 2  # BF16 = 2 bytes
        comp_nbytes = (
            self.sign_mantissa.nelement() * self.sign_mantissa.element_size()
            + self.compressed_full.nelement() * self.compressed_full.element_size()
            + self.bitmap1.nelement() * self.bitmap1.element_size()
            + self.bitmap2.nelement() * self.bitmap2.element_size()
            + self.bitmap3.nelement() * self.bitmap3.element_size()
            + self.tile_offsets.nelement() * self.tile_offsets.element_size()
            + self.tile_offsets_median.nelement() * self.tile_offsets_median.element_size()
            + self.tile_offsets_global.nelement() * self.tile_offsets_global.element_size()
        )
        return orig_nbytes / comp_nbytes

    def state_dict(self) -> dict:
        """Return a plain dict serializable with ``torch.save``."""
        return {
            "top_exponents": self.top_exponents,
            "sign_mantissa": self.sign_mantissa,
            "compressed_full": self.compressed_full,
            "bitmap1": self.bitmap1,
            "bitmap2": self.bitmap2,
            "bitmap3": self.bitmap3,
            "tile_offsets": self.tile_offsets,
            "tile_offsets_median": self.tile_offsets_median,
            "tile_offsets_global": self.tile_offsets_global,
            "max_high_freq_count": self.max_high_freq_count,
            "max_full_count": self.max_full_count,
            "num_global_tiles": self.num_global_tiles,
            "orig_shape": self.orig_shape,
        }

    @classmethod
    def from_state_dict(cls, d: dict):
        """Rebuild from a dict loaded with ``torch.load``."""
        return cls(
            top_exponents=d["top_exponents"],
            sign_mantissa=d["sign_mantissa"],
            compressed_full=d["compressed_full"],
            bitmap1=d["bitmap1"],
            bitmap2=d["bitmap2"],
            bitmap3=d["bitmap3"],
            tile_offsets=d["tile_offsets"],
            tile_offsets_median=d["tile_offsets_median"],
            tile_offsets_global=d["tile_offsets_global"],
            max_high_freq_count=d["max_high_freq_count"],
            max_full_count=d["max_full_count"],
            num_global_tiles=d["num_global_tiles"],
            orig_shape=d["orig_shape"],
        )

    def to(self, device, non_blocking=False):
        """Move all tensor fields to *device* and return self."""
        self.sign_mantissa = self.sign_mantissa.to(device, non_blocking=non_blocking)
        self.compressed_full = self.compressed_full.to(device, non_blocking=non_blocking)
        self.bitmap1 = self.bitmap1.to(device, non_blocking=non_blocking)
        self.bitmap2 = self.bitmap2.to(device, non_blocking=non_blocking)
        self.bitmap3 = self.bitmap3.to(device, non_blocking=non_blocking)
        self.tile_offsets = self.tile_offsets.to(device, non_blocking=non_blocking)
        self.tile_offsets_median = self.tile_offsets_median.to(device, non_blocking=non_blocking)
        self.tile_offsets_global = self.tile_offsets_global.to(device, non_blocking=non_blocking)
        self.top_exponents = self.top_exponents.to(device, non_blocking=non_blocking)
        return self

    def decompress(
        self, device: str = "cuda", output: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Reconstruct the original BF16 matrix on *device*.

        Parameters
        ----------
        device : str, optional
            Target torch device (default ``"cuda"``).
        output : torch.Tensor | None
            Optional pre-allocated output buffer of shape ``orig_shape`` and
            dtype ``torch.bfloat16`` on *device*. If not given, a new tensor is
            allocated.

        Returns
        -------
        torch.Tensor
            Reconstructed ``bfloat16`` tensor of shape ``orig_shape``.
        """
        import torch as _torch
        target = _torch.empty(1, device=device).device

        for name in (
            "sign_mantissa",
            "compressed_full",
            "bitmap1",
            "bitmap2",
            "bitmap3",
            "tile_offsets_median",
            "tile_offsets_global",
            "top_exponents",
        ):
            t = getattr(self, name)
            if t.device != target:
                setattr(self, name, t.to(target, non_blocking=True))

        M, K = self.orig_shape
        if output is None:
            output = _torch.empty((M, K), dtype=_torch.bfloat16, device=device)
        else:
            # sanity checks
            if output.shape != (M, K) or output.dtype != _torch.bfloat16:
                raise ValueError(
                    f"output shape/dtype mismatch: expected ({M}, {K}), "
                    f"bfloat16, got {output.shape}, {output.dtype}"
                )
            if output.device != target:
                raise ValueError(
                    f"output device mismatch: expected {target}, got {output.device}"
                )

        zipserv.kerenl_ops.bf16_decompress(
            self.sign_mantissa,
            self.compressed_full,
            self.bitmap1,
            self.bitmap2,
            self.bitmap3,
            self.tile_offsets_median,
            self.tile_offsets_global,
            self.max_high_freq_count,
            self.max_full_count,
            self.top_exponents,
            output,
            M,
            K,
        )
        return output


def compress(tensor: torch.Tensor) -> CompressedTensor:
    """Compress a BF16 CPU matrix and return a ``CompressedTensor``."""
    if tensor.dim() != 2:
        raise ValueError("compress expects a 2-D tensor")
    if tensor.dtype != torch.bfloat16:
        raise TypeError("compress expects dtype torch.bfloat16")
    if tensor.device.type != "cpu":
        raise ValueError("compress works on CPU tensors only")

    M, K = tensor.shape
    if M % 64 != 0 or K % 64 != 0:
        raise ValueError(
            f"tensor dimensions must be multiples of 64, got ({M}, {K})"
        )

    top_exp = _get_top_exponents(tensor)
    tup = zipserv.init_bf16_matrix_triple_bitmap(tensor, top_exponents=top_exp)
    return CompressedTensor._from_compress_tuple(
        (top_exp,) + tup,
        orig_shape=(M, K),
    )
