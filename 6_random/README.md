## Random
In the lecture, we learn how to use random numbers inside triton.jit'd function.


## Example: random initialize 1D tensor

In the following example, we initialize out tensor with random numbers.

```bash
import torch
import triton
import triton.language as tl

@triton.jit
def rand_init(out_ptr, n, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    random = tl.rand(seed, offsets)
    tl.store(out_ptr + offsets, random, mask=mask)


out = torch.zeros(size=(10,)).to('cuda')
print('initial out', out)
n= out.numel()

grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
rand_init[grid](out, n, seed=999, BLOCK_SIZE=2)
print('out with seed=999', out)

rand_init[grid](out, n, seed=999, BLOCK_SIZE=2)
print('out with seed=999', out)

rand_init[grid](out, n, seed=777, BLOCK_SIZE=2)
print('out with seed=777', out)
```

With the same seed 999, the out tensor is initialized with the same numbers.
With different seed 999 and 777, the out tensor is initialized with different numbers.
```bash
initial out tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], device='cuda:0')
out with seed=999 tensor([0.2196, 0.5715, 0.2051, 0.0709, 0.8743, 0.2240, 0.3336, 0.5628, 0.2430,
        0.9592], device='cuda:0')
out with seed=999 tensor([0.2196, 0.5715, 0.2051, 0.0709, 0.8743, 0.2240, 0.3336, 0.5628, 0.2430,
        0.9592], device='cuda:0')
out with seed=777 tensor([0.7059, 0.8081, 0.5869, 0.8980, 0.0776, 0.3504, 0.3932, 0.3695, 0.5784,
        0.5037], device='cuda:0')
```

## triton.language.rand()

tl.rand() function returns a block of random float32 in uniform random range [0, 1), where N_ROUNDS_DEFAULT=10.
The size of returned block is len(offset). The value of offset is used for retrieving different value for each offset position.
It first generate random integers and then convert it to float numbers.
```bash
@jit
def rand(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    offset = offset.to(tl.uint32, bitcast=True)
    source = randint(seed, offset, n_rounds)
    return uint32_to_uniform_float(source)
```

uin32_to_uniform_float() function converts a random uint32 into a random float uniformly sampled in [0, 1).
```bash
@jit
def uint32_to_uniform_float(x):
    x = x.to(tl.int32, bitcast=True)
    # maximum value such that `MAX_INT * scale < 1.0` (with float rounding)
    scale = 4.6566127342e-10
    x = tl.where(x < 0, -x - 1, x)
    return x * scale
```


randint() function returns a single block of random int32.
```bash
@jit
def randint(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    ret, _, _, _ = randint4x(seed, offset, n_rounds)
    return ret
```


randint4x() function returns four blocks of random int32.
```bash
@jit
def randint4x(seed, offset, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    _0 = offset * 0
    return philox(seed, offset, _0, _0, _0, n_rounds)
```

Philox is a counter-based pseudo random number generator.
```bash
@jit
def philox(seed, c0, c1, c2, c3, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    seed = seed.to(tl.uint64)
    seed_hi = ((seed >> 32) & 0xffffffff).to(tl.uint32)
    seed_lo = (seed & 0xffffffff).to(tl.uint32)
    c0 = c0.to(tl.uint32, bitcast=True)
    c1 = c1.to(tl.uint32, bitcast=True)
    c2 = c2.to(tl.uint32, bitcast=True)
    c3 = c3.to(tl.uint32, bitcast=True)
    return philox_impl(c0, c1, c2, c3, seed_lo, seed_hi, n_rounds)
```

philox_impl() function runs `n_rounds` rounds of Philox for state (c0, c1, c2, c3) and key (k0, k1).

```bash
PHILOX_KEY_A: tl.constexpr = 0x9E3779B9
PHILOX_KEY_B: tl.constexpr = 0xBB67AE85
PHILOX_ROUND_A: tl.constexpr = 0xD2511F53
PHILOX_ROUND_B: tl.constexpr = 0xCD9E8D57
@jit
def philox_impl(c0, c1, c2, c3, k0, k1, n_rounds: tl.constexpr = N_ROUNDS_DEFAULT):
    for _ in tl.static_range(n_rounds):
        # update random state
        A = PHILOX_ROUND_A
        B = PHILOX_ROUND_B
        _c0, _c2 = c0, c2
        c0 = tl.umulhi(B, _c2) ^ c1 ^ k0
        c2 = tl.umulhi(A, _c0) ^ c3 ^ k1
        c1 = B * _c2
        c3 = A * _c0
        # raise key
        k0 = k0 + PHILOX_KEY_A
        k1 = k1 + PHILOX_KEY_B
    return c0, c1, c2, c3
```

umulhi() function returns the most significant 32 bits of the product of x and y.
```bash
@builtin
def umulhi(x, y, _builder=None):
    x = _to_tensor(x, _builder)
    y = _to_tensor(y, _builder)
    return semantic.umulhi(x, y, _builder)
```
