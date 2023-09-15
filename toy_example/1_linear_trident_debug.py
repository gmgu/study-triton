import trident
import torch
import torch.nn as nn
import time

import triton
import triton.language as tl
from typing import Any
from trident import util
from trident.kernel import linear_configs

device = torch.device('cuda:0')

# This is the implementation of trident.Linear copied from kakaobrain/trident/trident/module.py
class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_accelerator: bool = False,
        device=None,
        dtype=None,
    ):
        """
        Applies Linear Transformation to an input.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            bias: If set to False, the layer will not learn an additive bias
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.use_accelerator = use_accelerator
        self.reset_parameters()

    def forward(self, input: torch.Tensor):
        """
        Args:
            input: an input (*, in_features)

        Returns:
            an output (*, out_features)
        """
        return function.linear(input, self.weight, self.bias, self.use_accelerator)

    def extra_repr(self):
        """
        Set the extra representation of the module.

        Returns:
            customized extra information
        """
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"backend=Trident"
        )

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = util.calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            util.uniform(self.bias, -bound, bound)

# implementation of trident.function.linear
# inside of this function, call static function Linear.apply of class Linear defined in trident.operation.linear,
# where trident.operation.linear.Linear is a subclass of torch.autograd.Function.
# Therefore, apply function will call Linear.forward(), and later loss.backward() will call Linear.backward()
def linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    use_accelerator: bool = False,
):
    """
    Applies Linear Transformation to an input.

    See Linear for more details.
    """
    if input.dim() == 2:
        output = operation.Linear.apply(input.view(1, *input.shape), weight, bias, use_accelerator)
        return output.view(output.shape[1:3])
    else:
        return operation.Linear.apply(input, weight, bias, use_accelerator)


# Here is a part of trident.operation.linear.Linear class
# (I didn't copy backward() and __backward())
class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any):
        input, weight, bias, use_accelerator = args

        util.push_trace("Linear.__forward")
        output = Linear.__forward(input, weight, bias, use_accelerator)
        util.pop_trace()

        ctx.save_for_backward(input, weight, bias, output)
        ctx.use_accelerator = use_accelerator

        return output

    @staticmethod
    def __forward(input, weight, bias, use_accelerator):
        factory_kwargs = {"device": input.device, "dtype": input.dtype}
        num_batches, m_size, k_size = input.shape
        n_size, _ = weight.shape
        output = torch.empty(num_batches, m_size, n_size, **factory_kwargs)

        def grid(meta):
            num_m_blocks = triton.cdiv(m_size, meta["m_block_size"])
            num_n_blocks = triton.cdiv(n_size, meta["n_block_size"])
            return (num_batches * num_m_blocks * num_n_blocks,)

        util.push_trace("kernel.Linear.forward")
        kernel.Linear.forward[grid](
            output,
            input,
            weight,
            bias,
            m_size,
            n_size,
            k_size,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            weight.stride(0),
            weight.stride(1),
            use_accelerator,
            util.dtype(input.dtype),
        )
        util.pop_trace()

        return output


# Here is a part of trident.kernel.Linear class
class Linear:
    @staticmethod
    @util.autotune(configs=linear_configs(), key=["m_size", "n_size", "k_size"])
    @triton.jit
    def forward(
        output_ptr: tl.tensor,
        input_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        bias_ptr: tl.tensor,
        m_size: tl.int32,
        n_size: tl.int32,
        k_size: tl.int32,
        input_batch_stride: tl.int32,
        input_m_stride: tl.int32,
        input_k_stride: tl.int32,
        weight_n_stride: tl.int32,
        weight_k_stride: tl.int32,
        use_accelerator: tl.constexpr,
        dtype: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_m_blocks = tl.cdiv(m_size, m_block_size)
        num_n_blocks = tl.cdiv(n_size, n_block_size)
        num_blocks = num_m_blocks * num_n_blocks
        batch = pid // num_blocks
        block = pid % num_blocks
        m_block = block // num_n_blocks
        n_block = block % num_n_blocks
        m_offset = m_block * m_block_size
        n_offset = n_block * n_block_size

        output = language.Linear.forward(
            input_ptr + batch * input_batch_stride,
            weight_ptr,
            bias_ptr,
            m_size,
            n_size,
            k_size,
            input_m_stride,
            input_k_stride,
            weight_n_stride,
            weight_k_stride,
            m_offset,
            n_offset,
            use_accelerator,
            m_block_size,
            n_block_size,
            k_block_size,
            dtype,
        )
        output_block_ptr = tl.make_block_ptr(
            output_ptr + batch * m_size * n_size,
            shape=(m_size, n_size),
            strides=(n_size, 1),
            offsets=(m_offset, n_offset),
            block_shape=(m_block_size, n_block_size),
            order=(1, 0),
        )
        tl.store(output_block_ptr, output, boundary_check=(0, 1))

# Here is a part of trident.language.Linear class
class Linear:
    @staticmethod
    @triton.jit
    def forward(
        input_ptr: tl.tensor,
        weight_ptr: tl.tensor,
        bias_ptr: tl.tensor,
        m_size: tl.int32,
        n_size: tl.int32,
        k_size: tl.int32,
        input_m_stride: tl.int32,
        input_k_stride: tl.int32,
        weight_n_stride: tl.int32,
        weight_k_stride: tl.int32,
        m_offset: tl.int32,
        n_offset: tl.int32,
        use_accelerator: tl.constexpr,
        m_block_size: tl.constexpr,
        n_block_size: tl.constexpr,
        k_block_size: tl.constexpr,
        dtype: tl.constexpr,
    ):
        input_block_ptr = tl.make_block_ptr(
            input_ptr,
            shape=(m_size, k_size),
            strides=(input_m_stride, input_k_stride),
            offsets=(m_offset, 0),
            block_shape=(m_block_size, k_block_size),
            order=(1, 0),
        )
        weight_block_ptr = tl.make_block_ptr(
            weight_ptr,
            shape=(k_size, n_size),
            strides=(weight_k_stride, weight_n_stride),
            offsets=(0, n_offset),
            block_shape=(k_block_size, n_block_size),
            order=(0, 1),
        )
        output = tl.zeros((m_block_size, n_block_size), dtype)

        for k_offset in range(0, k_size, k_block_size):
            input = tl.load(input_block_ptr, boundary_check=(0, 1), padding_option="zero")
            weight = tl.load(weight_block_ptr, boundary_check=(0, 1), padding_option="zero")
            output += tl.dot(input, weight, use_accelerator).to(dtype)
            input_block_ptr = tl.advance(input_block_ptr, (0, k_block_size))
            weight_block_ptr = tl.advance(weight_block_ptr, (k_block_size, 0))

        if bias_ptr is not None:
            bias_block_ptr = tl.make_block_ptr(
                bias_ptr,
                shape=(n_size,),
                strides=(1,),
                offsets=(n_offset,),
                block_shape=(n_block_size,),
                order=(0,),
            )
            bias = tl.load(bias_block_ptr, boundary_check=(0,), padding_option="zero")
            output += bias

        return output

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = trident.Linear(10, 1, use_accelerator=True)
        #print('weight', self.layer.weight)
        #print('bias', self.layer.bias)
        #print('use_accelerator', self.layer.use_accelerator)
        #print(self.layer)  # call for extra_repr

    
    def forward(self, x):
        #print('x.dim():', x.dim())
        #print('x.shape:', x.shape)
        x = self.layer(x)

        return x


model = LinearModel().to(device)

data_x = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(device)

# for compile
pred_y = model(data_x)

start_time = time.time()
for _ in range(1_000_000):
    pred_y = model(data_x)
end_time = time.time()

print(pred_y)
print(f'Elapsed Time: {end_time - start_time:.2f} sec')
