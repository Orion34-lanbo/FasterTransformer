#include <torch/extension.h>
#include <torch/csrc/cuda/nccl.h>
#include <c10/cuda/CUDAGuard.h>

#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace torch_ext {

ncclComm_t ftNcclInitTensorParallel(int64_t world_size,
                                    int64_t world_rank,
                                    char* unique_id_char) {
#if 1
    ncclUniqueId tp_uid;
    for (int i = 0; i < 128; i++) {
      tp_uid.internal[i] = unique_id_char[i];
      if (world_rank == 0) {
          std::cout << i << " " << tp_uid.internal[i] << " " << unique_id_char[i] << std::endl;
      }
    }
    ncclComm_t tp_nccl_comm;
    std::cout << "unique id " << tp_uid.internal << " nrank " << world_size << " rank " << world_rank << std::endl;
    NCCLCHECK(ncclCommInitRank(&tp_nccl_comm, world_size, tp_uid, world_rank));
    return tp_nccl_comm;
#else
    ft::NcclParam tp, pp;
    ft::ftNcclInitialize(tp, pp, world_size, 1);
    return tp.nccl_comm_;
#endif
}

torch::Tensor all_reduce(torch::Tensor input,
                         torch::Tensor unique_id,
                         int64_t world_size,
                         int64_t world_rank) {
    static ncclComm_t tensor_para =
        ftNcclInitTensorParallel(world_size, world_rank, static_cast<char*>(unique_id.data_ptr()));
    auto stream = at::cuda::getDefaultCUDAStream();
    // auto comms = torch::cuda::nccl::detail::get_communicators(std::vector<torch::Tensor>(1, input));
    // int device = input.device().index();
#if 1
    ncclDataType_t nccl_data_type = ncclHalf;
    NCCLCHECK(ncclGroupStart());
    // at::cuda::OptionalCUDAGuard device_guard;
    // device_guard.set_index(device);
    NCCLCHECK(ncclAllReduce(
        input.data_ptr(),
        input.data_ptr(),
        input.numel(),
        nccl_data_type,
        ncclSum,
        // reinterpret_cast<ncclComm_t>(comms[0]),
        tensor_para,
        stream));
    NCCLCHECK(ncclGroupEnd());
    // sync_check_cuda_error();
    // cudaDeviceSynchronize();
    // cudaStreamSynchronize(stream);
#else
    std::cout << "nrank " << world_size
              << " rank " << world_rank
              << " device " << device
              << " data type " << input.dtype()
              << " num elements " << input.numel()
              // << " comm " << comms[0]
              << std::endl;
    ft::ftNcclAllReduceSum(static_cast<half*>(input.data_ptr()),
                           static_cast<half*>(input.data_ptr()),
                           input.nbytes(),
                           nccl_param,
                           at::cuda::getCurrentCUDAStream());
#endif
    return input;
}

}  // namespace torch_ext

TORCH_LIBRARY(megatron, m) {
  m.def("all_reduce", torch_ext::all_reduce);
}
