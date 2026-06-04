#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include "L_API.cuh"

namespace py = pybind11;

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------
static inline void check_cuda_dtype(const torch::Tensor& t,
                                    const char* name,
                                    at::ScalarType dtype) {
  TORCH_CHECK(t.device().is_cuda(),
              name, " must be a CUDA tensor, got ", t.device());
  TORCH_CHECK(t.scalar_type() == dtype,
              name, " must have scalar type ", dtype, ", got ", t.scalar_type());
}

static inline void check_cpu_dtype(const torch::Tensor& t,
                                   const char* name,
                                   at::ScalarType dtype) {
  TORCH_CHECK(t.device().is_cpu(),
              name, " must be a CPU tensor, got ", t.device());
  TORCH_CHECK(t.scalar_type() == dtype,
              name, " must have scalar type ", dtype, ", got ", t.scalar_type());
}

// ------------------------------------------------------------------
// 1. BF16 Triple Bitmap Matrix Multiplication
// ------------------------------------------------------------------
void bf16_matmul(
    const torch::Tensor& sign_mantissa,       // uint8, CUDA
    const torch::Tensor& compressed_full,     // bfloat16, CUDA
    const torch::Tensor& bitmap1,               // int64, CUDA
    const torch::Tensor& bitmap2,             // int64, CUDA
    const torch::Tensor& bitmap3,             // int64, CUDA
    const torch::Tensor& tile_offsets_median, // int32, CUDA
    const torch::Tensor& tile_offsets_global, // int32, CUDA
    int max_high_freq_count,
    int max_full_count,
    const torch::Tensor& top_exponents,         // int32, CUDA, 7 elements
    const torch::Tensor& B,                   // bfloat16, CUDA
    torch::Tensor C,                          // bfloat16, CUDA (out)
    py::object reduction_workspace_obj,         // bfloat16, CUDA or None
    int split_k,
    int M_Global,
    int N_Global,
    int K_Global)
{
  check_cuda_dtype(sign_mantissa,       "sign_mantissa",       at::ScalarType::Byte);
  check_cuda_dtype(compressed_full,     "compressed_full",     at::ScalarType::BFloat16);
  check_cuda_dtype(bitmap1,             "bitmap1",             at::ScalarType::Long);
  check_cuda_dtype(bitmap2,             "bitmap2",             at::ScalarType::Long);
  check_cuda_dtype(bitmap3,             "bitmap3",             at::ScalarType::Long);
  check_cuda_dtype(tile_offsets_median, "tile_offsets_median", at::ScalarType::Int);
  check_cuda_dtype(tile_offsets_global, "tile_offsets_global", at::ScalarType::Int);
  check_cuda_dtype(B,                   "B",                   at::ScalarType::BFloat16);
  check_cuda_dtype(C,                   "C",                   at::ScalarType::BFloat16);
  check_cuda_dtype(top_exponents,       "top_exponents",       at::ScalarType::Int);
  TORCH_CHECK(top_exponents.numel() == 7,
              "top_exponents must contain exactly 7 elements");

  __nv_bfloat16* reduction_ptr = nullptr;
  if (!reduction_workspace_obj.is_none()) {
    torch::Tensor reduction_workspace = reduction_workspace_obj.cast<torch::Tensor>();
    if (split_k > 1) {
      check_cuda_dtype(reduction_workspace, "reduction_workspace", at::ScalarType::BFloat16);
    }
    reduction_ptr = reinterpret_cast<__nv_bfloat16*>(
        reduction_workspace.data_ptr<c10::BFloat16>());
  } else if (split_k > 1) {
    throw std::invalid_argument("reduction_workspace is required when split_k > 1");
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  cudaError_t err = BF16TripleBitmap_MM_API(
      stream,
      sign_mantissa.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(compressed_full.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const uint64_t*>(bitmap1.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(bitmap2.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(bitmap3.data_ptr<int64_t>()),
      tile_offsets_median.data_ptr<int32_t>(),
      tile_offsets_global.data_ptr<int32_t>(),
      max_high_freq_count,
      max_full_count,
      top_exponents.data_ptr<int32_t>(),
      reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<c10::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(C.data_ptr<c10::BFloat16>()),
      M_Global,
      N_Global,
      K_Global,
      reduction_ptr,
      split_k);

  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("BF16TripleBitmap_MM_API failed: ") + cudaGetErrorString(err));
  }
}

// ------------------------------------------------------------------
// 2. BF16 Triple Bitmap Decompress
// ------------------------------------------------------------------
void bf16_decompress(
    const torch::Tensor& sign_mantissa,
    const torch::Tensor& compressed_full,
    const torch::Tensor& bitmap1,
    const torch::Tensor& bitmap2,
    const torch::Tensor& bitmap3,
    const torch::Tensor& tile_offsets_median,
    const torch::Tensor& tile_offsets_global,
    int max_high_freq_count,
    int max_full_count,
    const torch::Tensor& top_exponents,
    torch::Tensor output,
    int M_Global,
    int K_Global,
    std::optional<torch::Tensor> profile_buffer = std::nullopt)
{
  check_cuda_dtype(sign_mantissa,       "sign_mantissa",       at::ScalarType::Byte);
  check_cuda_dtype(compressed_full,     "compressed_full",     at::ScalarType::BFloat16);
  check_cuda_dtype(bitmap1,             "bitmap1",             at::ScalarType::Long);
  check_cuda_dtype(bitmap2,             "bitmap2",             at::ScalarType::Long);
  check_cuda_dtype(bitmap3,             "bitmap3",             at::ScalarType::Long);
  check_cuda_dtype(tile_offsets_median, "tile_offsets_median", at::ScalarType::Int);
  check_cuda_dtype(tile_offsets_global, "tile_offsets_global", at::ScalarType::Int);
  check_cuda_dtype(output,              "output",              at::ScalarType::BFloat16);
  check_cuda_dtype(top_exponents,       "top_exponents",       at::ScalarType::Int);
  TORCH_CHECK(top_exponents.numel() == 7,
              "top_exponents must contain exactly 7 elements");

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  uint64_t* profile_ptr = nullptr;
  int profile_enabled = 0;
  if (profile_buffer.has_value()) {
    auto buf = profile_buffer.value();
    check_cuda_dtype(buf, "profile_buffer", at::ScalarType::Long);
    profile_ptr = reinterpret_cast<uint64_t*>(buf.data_ptr<int64_t>());
    profile_enabled = 1;
  }

  cudaError_t err = BF16TripleBitmap_Decompress_API(
      stream,
      sign_mantissa.data_ptr<uint8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(compressed_full.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const uint64_t*>(bitmap1.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(bitmap2.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(bitmap3.data_ptr<int64_t>()),
      tile_offsets_median.data_ptr<int32_t>(),
      tile_offsets_global.data_ptr<int32_t>(),
      top_exponents.data_ptr<int32_t>(),
      max_high_freq_count,
      max_full_count,
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<c10::BFloat16>()),
      M_Global,
      K_Global,
      profile_ptr,
      profile_enabled);

  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("BF16TripleBitmap_Decompress_API failed: ") + cudaGetErrorString(err));
  }
}

// ------------------------------------------------------------------
// 3. Init BF16 Matrix Triple Bitmap (CPU-side)
// ------------------------------------------------------------------
std::tuple<
    torch::Tensor, // sign_mantissa (uint8 CPU)
    torch::Tensor, // compressed_full (bf16 CPU)
    torch::Tensor, // bitmap1 (int64 CPU)
    torch::Tensor, // bitmap2
    torch::Tensor, // bitmap3
    torch::Tensor, // tile_offsets (int32 CPU)
    torch::Tensor, // tile_offsets_median
    torch::Tensor, // tile_offsets_global
    int,           // max_high_freq_count
    int,           // max_full_count
    int            // num_global_tiles
>
init_bf16_matrix_triple_bitmap(
    const torch::Tensor& A_bf16,    // bfloat16, CPU, 2D [M, K]
    int tile_M,
    int tile_M_median,
    int tile_M_global,
    int tile_K,
    int tile_K_median,
    int tile_K_global,
    const torch::Tensor& top_exponents) // int32, CPU, 7 elements
{
  check_cpu_dtype(A_bf16, "A_bf16", at::ScalarType::BFloat16);
  TORCH_CHECK(A_bf16.dim() == 2, "A_bf16 must be 2D");
  int M = static_cast<int>(A_bf16.size(0));
  int K = static_cast<int>(A_bf16.size(1));

  check_cpu_dtype(top_exponents, "top_exponents", at::ScalarType::Int);
  TORCH_CHECK(top_exponents.numel() == 7,
              "top_exponents must contain exactly 7 elements");

  uint8_t* sign_mantissa = nullptr;
  __nv_bfloat16* compressed_full = nullptr;
  uint64_t* bitmap1 = nullptr;
  uint64_t* bitmap2 = nullptr;
  uint64_t* bitmap3 = nullptr;
  int* tile_offsets = nullptr;
  int* tile_offsets_median = nullptr;
  int* tile_offsets_global = nullptr;

  int max_high_freq_count = 0;
  int max_full_count = 0;

  int num_global_tiles = InitBF16MatrixTripleBitmap(
      reinterpret_cast<__nv_bfloat16*>(A_bf16.data_ptr<c10::BFloat16>()),
      M, K,
      tile_M, tile_M_median, tile_M_global,
      tile_K, tile_K_median, tile_K_global,
      top_exponents.data_ptr<int32_t>(),
      &sign_mantissa,
      &compressed_full,
      &bitmap1,
      &bitmap2,
      &bitmap3,
      &tile_offsets,
      &tile_offsets_median,
      &tile_offsets_global,
      max_high_freq_count,
      max_full_count);

  if (num_global_tiles < 0) {
    throw std::runtime_error("InitBF16MatrixTripleBitmap failed (memory allocation error)");
  }

  // Actual sizes are known from the cumulative counts at the end of the global offsets.
  int total_high_freq = tile_offsets_global[num_global_tiles * 2];
  int total_full      = tile_offsets_global[num_global_tiles * 2 + 1];

  int num_tiles = (M / tile_M) * (K / tile_K);
  int num_median_tiles = (M / tile_M_median) * (K / tile_K_median);

  auto free_deleter = [](void* p) { free(p); };

  torch::Tensor t_sign_mantissa = torch::from_blob(
      sign_mantissa,
      {total_high_freq},
      free_deleter,
      torch::TensorOptions().dtype(at::ScalarType::Byte));

  torch::Tensor t_compressed_full = torch::from_blob(
      compressed_full,
      {total_full},
      free_deleter,
      torch::TensorOptions().dtype(at::ScalarType::BFloat16));

  torch::Tensor t_bitmap1 = torch::from_blob(
      reinterpret_cast<void*>(bitmap1),
      {num_tiles},
      free_deleter,
      torch::TensorOptions().dtype(at::ScalarType::Long));

  torch::Tensor t_bitmap2 = torch::from_blob(
      reinterpret_cast<void*>(bitmap2),
      {num_tiles},
      free_deleter,
      torch::TensorOptions().dtype(at::ScalarType::Long));

  torch::Tensor t_bitmap3 = torch::from_blob(
      reinterpret_cast<void*>(bitmap3),
      {num_tiles},
      free_deleter,
      torch::TensorOptions().dtype(at::ScalarType::Long));

  torch::Tensor t_tile_offsets = torch::from_blob(
      tile_offsets,
      {num_tiles, 2},
      free_deleter,
      torch::TensorOptions().dtype(at::ScalarType::Int));

  torch::Tensor t_tile_offsets_median = torch::from_blob(
      tile_offsets_median,
      {num_median_tiles, 2},
      free_deleter,
      torch::TensorOptions().dtype(at::ScalarType::Int));

  torch::Tensor t_tile_offsets_global = torch::from_blob(
      tile_offsets_global,
      {num_global_tiles + 1, 2},
      free_deleter,
      torch::TensorOptions().dtype(at::ScalarType::Int));

  return std::make_tuple(
      t_sign_mantissa,
      t_compressed_full,
      t_bitmap1,
      t_bitmap2,
      t_bitmap3,
      t_tile_offsets,
      t_tile_offsets_median,
      t_tile_offsets_global,
      max_high_freq_count,
      max_full_count,
      num_global_tiles);
}

// ------------------------------------------------------------------
// Module definition
// ------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "ZipServ kernel operations";

  m.def("bf16_matmul", &bf16_matmul,
          "BF16 Triple Bitmap Matrix Multiplication",
          py::arg("sign_mantissa"),
          py::arg("compressed_full"),
          py::arg("bitmap1"),
          py::arg("bitmap2"),
          py::arg("bitmap3"),
          py::arg("tile_offsets_median"),
          py::arg("tile_offsets_global"),
          py::arg("max_high_freq_count"),
          py::arg("max_full_count"),
           py::arg("top_exponents"),
          py::arg("B"),
          py::arg("C"),
          py::arg("reduction_workspace") = py::none(),
          py::arg("split_k") = 1,
          py::arg("M"),
          py::arg("N"),
          py::arg("K"));

  m.def("bf16_decompress", &bf16_decompress,
          "BF16 Triple Bitmap Decompress",
          py::arg("sign_mantissa"),
          py::arg("compressed_full"),
          py::arg("bitmap1"),
          py::arg("bitmap2"),
          py::arg("bitmap3"),
          py::arg("tile_offsets_median"),
          py::arg("tile_offsets_global"),
          py::arg("max_high_freq_count"),
          py::arg("max_full_count"),
           py::arg("top_exponents"),
          py::arg("output"),
          py::arg("M"),
          py::arg("K"),
          py::arg("profile_buffer") = py::none());

  m.def("init_bf16_matrix_triple_bitmap", &init_bf16_matrix_triple_bitmap,
          "Initialize BF16 Triple Bitmap Compression on CPU",
          py::arg("A_bf16"),
          py::arg("tile_M") = 8,
          py::arg("tile_M_median") = 16,
          py::arg("tile_M_global") = 64,
          py::arg("tile_K") = 8,
          py::arg("tile_K_median") = 64,
          py::arg("tile_K_global") = 64,
          py::arg("top_exponents"));
}
