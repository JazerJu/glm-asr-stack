#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" {

int flash_fwd_launch_varlen(
    void *q_ptr, void *k_ptr, void *v_ptr, void *out_ptr,
    int seqlen, int nheads, int head_dim,
    float softmax_scale,
    CUstream ext_stream)
{
    try {
        auto device = torch::Device(torch::kCUDA, 0);
        auto dtype = torch::kBFloat16;
        int hidden = nheads * head_dim;
        auto options = torch::TensorOptions().dtype(dtype).device(device);
        
        auto q = torch::from_blob(q_ptr, {seqlen, nheads, head_dim}, 
            {hidden, head_dim, 1}, options);
        auto k = torch::from_blob(k_ptr, {seqlen, nheads, head_dim},
            {hidden, head_dim, 1}, options);
        auto v = torch::from_blob(v_ptr, {seqlen, nheads, head_dim},
            {hidden, head_dim, 1}, options);
        
        auto cu = torch::tensor({0, seqlen}, 
            torch::TensorOptions().dtype(torch::kInt32).device(device));
        auto out = torch::from_blob(out_ptr, {seqlen, nheads, head_dim},
            {hidden, head_dim, 1}, options);
        
        if (softmax_scale <= 0.0f)
            softmax_scale = 1.0f / sqrtf((float)head_dim);
        
        at::cuda::CUDAStream torch_stream = 
            at::cuda::getStreamFromExternal(ext_stream, device.index());
        at::cuda::setCurrentCUDAStream(torch_stream);
        
        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("_vllm_fa2_C::varlen_fwd", "");
        
        c10::Stack stack;
        stack.reserve(21);
        stack.push_back(q);
        stack.push_back(k);
        stack.push_back(v);
        stack.push_back(out);
        stack.push_back(cu);
        stack.push_back(cu);
        stack.push_back(c10::IValue());  // seqused_k
        stack.push_back(c10::IValue());  // leftpad_k
        stack.push_back(c10::IValue());  // block_table
        stack.push_back(c10::IValue());  // alibi_slopes
        stack.push_back(c10::IValue(seqlen));
        stack.push_back(c10::IValue(seqlen));
        stack.push_back(c10::IValue(0.0));
        stack.push_back(c10::IValue((double)softmax_scale));
        stack.push_back(c10::IValue(false));
        stack.push_back(c10::IValue(false));
        stack.push_back(c10::IValue(int64_t(-1)));
        stack.push_back(c10::IValue(int64_t(-1)));
        stack.push_back(c10::IValue(0.0));
        stack.push_back(c10::IValue(false));
        stack.push_back(c10::IValue(int64_t(1)));
        stack.push_back(c10::IValue());  // generator (None)
        
        op.callBoxed(stack);
        
        return 0;
    } catch (const c10::Error& e) {
        fprintf(stderr, "flash_fwd_bridge error: %s\n", e.what());
        return -1;
    } catch (const std::exception& e) {
        fprintf(stderr, "flash_fwd_bridge error: %s\n", e.what());
        return -1;
    }
}

int flash_fwd_launch_varlen_strided(
    void *q_ptr, void *k_ptr, void *v_ptr, void *out_ptr,
    int seqlen, int nheads, int head_dim,
    float softmax_scale,
    int q_stride, int k_stride, int v_stride,
    CUstream ext_stream)
{
    try {
        auto device = torch::Device(torch::kCUDA, 0);
        auto dtype = torch::kBFloat16;
        int hidden = nheads * head_dim;
        auto options = torch::TensorOptions().dtype(dtype).device(device);

        auto q = torch::from_blob(q_ptr, {seqlen, nheads, head_dim},
            {q_stride, head_dim, 1}, options);
        auto k = torch::from_blob(k_ptr, {seqlen, nheads, head_dim},
            {k_stride, head_dim, 1}, options);
        auto v = torch::from_blob(v_ptr, {seqlen, nheads, head_dim},
            {v_stride, head_dim, 1}, options);
        auto out = torch::from_blob(out_ptr, {seqlen, nheads, head_dim},
            {hidden, head_dim, 1}, options);

        auto cu = torch::tensor({0, seqlen},
            torch::TensorOptions().dtype(torch::kInt32).device(device));

        if (softmax_scale <= 0.0f)
            softmax_scale = 1.0f / sqrtf((float)head_dim);

        at::cuda::CUDAStream torch_stream =
            at::cuda::getStreamFromExternal(ext_stream, device.index());
        at::cuda::setCurrentCUDAStream(torch_stream);

        static auto op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("_vllm_fa2_C::varlen_fwd", "");

        c10::Stack stack;
        stack.reserve(21);
        stack.push_back(q);
        stack.push_back(k);
        stack.push_back(v);
        stack.push_back(out);
        stack.push_back(cu);
        stack.push_back(cu);
        stack.push_back(c10::IValue());
        stack.push_back(c10::IValue());
        stack.push_back(c10::IValue());
        stack.push_back(c10::IValue());
        stack.push_back(c10::IValue(seqlen));
        stack.push_back(c10::IValue(seqlen));
        stack.push_back(c10::IValue(0.0));
        stack.push_back(c10::IValue((double)softmax_scale));
        stack.push_back(c10::IValue(false));
        stack.push_back(c10::IValue(false));
        stack.push_back(c10::IValue(int64_t(-1)));
        stack.push_back(c10::IValue(int64_t(-1)));
        stack.push_back(c10::IValue(0.0));
        stack.push_back(c10::IValue(false));
        stack.push_back(c10::IValue(int64_t(1)));
        stack.push_back(c10::IValue());

        op.callBoxed(stack);
        return 0;
    } catch (const c10::Error& e) {
        fprintf(stderr, "flash_fwd_bridge error: %s\n", e.what());
        return -1;
    } catch (const std::exception& e) {
        fprintf(stderr, "flash_fwd_bridge error: %s\n", e.what());
        return -1;
    }
}

} // extern "C"
