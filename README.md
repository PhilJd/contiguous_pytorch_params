# Contiguous Parameters for Pytorch.
Launching Cuda kernels comes with a small overhead, resulting in low GPU utilization
when launching numerous fast-returning kernels.
This repo accelerates training by copying all parameters into one contiguous
buffer, resetting the parameters to be views into the buffer, and applying
optimizer updates on the contiguous representation. Depending on the model and 
optimizer, this can drastically reduce the time required for the optimizer's step
function by a factor of 40x-100x.

For this to work, two requirements need to be fulfilled:
1. The computation graph may only alter the parameters and gradients inplace.
   Make sure to call `parameters.assert_buffer_is_valid()` to detect any buffer
   invalidation.
2. The operation executed on `parameters.contiguous()` must not rely on shape
   information or statistics of the parameter as these would be computed on the
   full buffer instead of each of the original parameters. For such operations,
   keep using `parameters.original()`.

## Install
```
pip install git+https://github.com/philjd/contiguous_pytorch_params.git
```

## Example Usage
```python
import torch
from torch import nn
from contiguous_params import ContiguousParams

data = torch.randn(5, 1, 8)
model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))

# Create the contiguous parameters.
parameters = ContiguousParams(model.parameters())

# Use parameters.contiguous() instead of model.parameters() to initialize
# the optimizer. Note that the optimize must update the parameters inplace.
optimizer = torch.optim.Adam(parameters.contiguous())

# Run the training loop as usual.
for x in data:
    loss = model(x).sum()
    loss.backward()
    # Gradient clipping also profits from contiguous memory.
    nn.utils.clip_grad_norm_(parameters.contiguous(), 0.1)
    optimizer.step()
    optimizer.zero_grad()
    # !!!!!!!
    # Always make sure to call buffer_is_valid() at least once, to detect
    # if operations invalidated the buffer by overwriting/copying parameters.
    # !!!!!!!
    parameters.assert_buffer_is_valid()
``` 


## Testing
```
pytest test.py
```

## Benchmarking
Run `python benchmark.py`. This applies several updates with the original method
as well as using contiguous parameters. You should see a speed up of ~100x.
To take a look at the timeline, open chromium, navigate to `chrome://tracing/`,
click load, and select the `*timeline.json` file.


## TODO
- [ ] Distributed training
