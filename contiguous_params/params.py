import torch
from torch import nn


class ContiguousParams:

    def __init__(self, parameters):
        # Create a list of the parameters to prevent emptying an iterator.
        self._parameters = list(parameters)
        self._param_buffer = None
        self._grad_buffer = None
        self._init_buffers()
        # Store the data pointers for each parameter into the buffer. These
        # can be used to check if an operation overwrites the gradient/data
        # tensor (invalidating the assumption of a contiguous buffer).
        self.data_pointers = []
        self.grad_pointers = []
        self.make_params_contiguous()

    def _init_buffers(self):
        dtype = self._parameters[0].dtype
        device = self._parameters[0].device
        if not all(p.dtype == dtype for p in self._parameters):
            raise ValueError("All parameters must be of the same dtype.")
        if not all(p.device == device for p in self._parameters):
            raise ValueError("All parameters must be on the same device.")
        size = sum(p.numel() for p in self._parameters)
        self._param_buffer = torch.zeros(size, dtype=dtype, device=device)
        self._grad_buffer =  torch.zeros(size, dtype=dtype, device=device)

    def make_params_contiguous(self):
        """Create a buffer to hold all params and update the params to be views of the buffer.

        Args:
            parameters: An iterable of parameters.
        """
        index = 0
        for p in self._parameters:
            size = p.numel()
            self._param_buffer[index:index + size] = p.data.view(-1)
            p.data = self._param_buffer[index:index + size].view(p.data.shape)
            p.grad = self._grad_buffer[index:index + size].view(p.data.shape)
            self.data_pointers.append(p.data.data_ptr())
            self.grad_pointers.append(p.grad.data.data_ptr())
            index += size
        # Bend the param_buffer to use grad_buffer to track its gradients.
        self._param_buffer.grad = self._grad_buffer

    def contiguous(self):
        """Return all parameters as one contiguous buffer."""
        return [self._param_buffer]

    def original(self):
        """Return the non-flattened parameters."""
        return self._parameters

    def buffer_is_valid(self):
        """Verify that all parameters and gradients still use the buffer."""
        params_and_pointers = zip(self._parameters,
                                  self.data_pointers,
                                  self.grad_pointers)
        return all((p.data.data_ptr() == data_ptr) and
                   (p.grad.data.data_ptr() == grad_ptr)
                   for p, data_ptr, grad_ptr in params_and_pointers)

    def assert_buffer_is_valid(self):
        if not self.buffer_is_valid():
            raise ValueError(
                "The data or gradient buffer has been invalidated. Please make "
                "sure to use inplace operations only when updating parameters "
                "or gradients.")
