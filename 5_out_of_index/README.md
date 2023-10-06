## Control the out of index problem

## triton.cdiv
If a row (or column) of length n is divided by BLOCK_SIZE, we set the number of threads to n / BLOCK_SIZE. But if a row is not divided by BLOCK_SIZE, we need to set the number of threads to n / BLOCK_SIZE + 1. triton.cdiv does exactly this job.

```bash
def cdiv(x: int, y: int):
    return (x + y - 1) // y
```

If x % y == 0, then x = k * y, where k is a positive integer. triton.cdiv then returns (x + y - 1) // y = ((k + 1) * y - 1) // y = k.

If x % y != 0, then x = (k - 1) * y + z, where k and z are two positive integers such that 2 <= k and 1 <= z < y. triton.cdiv then returns (x + y - 1) // y = (k * y + z - 1) // y. Since 1 <= z < y, we have 0 <= z - 1 < y - 1. Thus, (k * y + z - 1) // y = k. (Note that -1 is useful for handling the first case: x % y == 0, and does not affect the second case: x % y != 0).

