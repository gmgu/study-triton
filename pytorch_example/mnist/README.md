# Basic MNIST Example

```bash
pip install -r requirements.txt
python main_trident.py  # for running PyTorch + Trident implementation
# python main_pytorch.py  # for running pure PyTorch implementation
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```


hmm... its not working.

using torch 1.13.1, triton 2.1.0, trident 0.1.1

The initialization is way too slow, and there is an error while running.

...

Train Epoch: 2 [59520/60000 (99%)]	Loss: 0.044929
ERROR: Unexpected segmentation fault encountered in worker.
Traceback (most recent call last):
  File "/home/gmgu/.local/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1120, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
  File "/usr/lib/python3.8/queue.py", line 179, in get
    self.not_empty.wait(remaining)
  File "/usr/lib/python3.8/threading.py", line 306, in wait
    gotit = waiter.acquire(True, timeout)
  File "/home/gmgu/.local/lib/python3.8/site-packages/torch/utils/data/_utils/signal_handling.py", line 66, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 20532) is killed by signal: Segmentation fault.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "main_trident.py", line 147, in <module>
    main()
  File "main_trident.py", line 139, in main
    test(model, device, test_loader)
  File "main_trident.py", line 61, in test
    for data, target in test_loader:
  File "/home/gmgu/.local/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 628, in __next__
    data = self._next_data()
  File "/home/gmgu/.local/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1316, in _next_data
    idx, data = self._get_data()
  File "/home/gmgu/.local/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1272, in _get_data
    success, data = self._try_get_data()
  File "/home/gmgu/.local/lib/python3.8/site-packages/torch/utils/data/dataloader.py", line 1133, in _try_get_data
    raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
RuntimeError: DataLoader worker (pid(s) 20532) exited unexpectedly
