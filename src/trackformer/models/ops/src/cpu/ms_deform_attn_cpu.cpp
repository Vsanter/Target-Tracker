#include <vector>

#include <ATen/ATen.h>

// CPU 版本暂时返回一个空结果，用于 CPU 模式编译
at::Tensor
ms_deform_attn_cpu_forward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    // 简单实现：返回输入 value（不做处理）
    // 注意：这只是占位实现，真正计算需要 CUDA
    return value.clone();
}

std::vector<at::Tensor>
ms_deform_attn_cpu_backward(
    const at::Tensor &value,
    const at::Tensor &spatial_shapes,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    // 简单实现：返回空梯度
    // 注意：这只是占位实现，真正计算需要 CUDA
    return {grad_output.clone(), grad_output.clone(), grad_output.clone(), grad_output.clone()};
}

