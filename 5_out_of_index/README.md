## Control the out of index problem

## triton.cdiv
If a row (or column) of length n is divided by BLOCK_SIZE, we set the number of threads to n / BLOCK_SIZE. But if a row is not divided by BLOCK_SIZE, we need to set the number of threads to n / BLOCK_SIZE + 1. triton.cdiv does exactly this job.

```bash
def cdiv(x: int, y: int):
    return (x + y - 1) // y
```

If x % y == 0, then x = k * y, where k is a positive integer. triton.cdiv then returns (x + y - 1) // y = ((k + 1) * y - 1) // y = k.

If x % y != 0, then x = (k - 1) * y + z, where k and z are two positive integers such that 2 <= k and 1 <= z < y. triton.cdiv then returns (x + y - 1) // y = (k * y + z - 1) // y. Since 1 <= z < y, we have 0 <= z - 1 < y - 1. Thus, (k * y + z - 1) // y = k. (Note that -1 is useful for handling the first case: x % y == 0, and does not affect the second case: x % y != 0).

## broadcasting
Broadcasting semantics.
1. Padding: the shape of the shortest operand is left-padded with ones until both operands have the same dimensionality.
2. Broadcasting: the content of both operands is replicated as many times as neede until their shape is identical; an error is emitted if this cannot be done.

```bash
int a[2] = {1, 2}
int b[4, 2] = {{3, 4}, {5, 6}, {7, 8}, {9, 10}}

int c[4, 2] = a + b
// 1. the shape of the a is left-padded with ones: a[1, 2] = {{1, 2}}
// 2. the content of a is replicated: a[4, 2] = {{1, 2}, {1, 2}, {1, 2}, {1, 2}}

```

