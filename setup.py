import os
import sys
from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    import torch
except ImportError as exc:
    print(
        "Error: PyTorch is required to build this package. "
        "Please install torch first (e.g., pip install torch).",
        file=sys.stderr,
    )
    raise

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------
SOURCES = [
    "csrc/L_API.cu",
    "csrc/zipserv_api.cpp",
]

INCLUDE_DIRS = [
    "csrc/",
]

# ------------------------------------------------------------------------------
# Compiler flags (mirror the original Makefile)
# ------------------------------------------------------------------------------
EXTRA_COMPILE_ARGS = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "--threads", "0",
        "-maxrregcount=255",
        "--use_fast_math",
        "--ptxas-options=-v,-warn-lmem-usage,--warn-on-spills",
        "-gencode", "arch=compute_80,code=sm_80",
        "-gencode", "arch=compute_86,code=sm_86",
        "-gencode", "arch=compute_89,code=sm_89",
    ],
}

LIBRARIES = ["cublas"]

TORCH_LIB_PATH = os.path.join(os.path.dirname(torch.__file__), "lib")
EXTRA_LINK_ARGS = [f"-Wl,-rpath,{TORCH_LIB_PATH}"]

# ------------------------------------------------------------------------------
# Extension module
# ------------------------------------------------------------------------------
ext_modules = [
    CUDAExtension(
        name="zipserv.kerenl_ops",
        sources=SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        libraries=LIBRARIES,
    )
]

setup(
    name="zipserv",
    version="0.1.0",
    description="ZipServ CUDA kernels with PyTorch bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=["zipserv"],
    zip_safe=False,
)
