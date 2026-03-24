import torch
from ml_core.storage import _tensor_to_b64, _b64_to_tensor
# _tensor_to_b64 and _b64_to_tensor are local so they don't require supabase

A = torch.rand(4, 4)
b = torch.rand(4, 1)

A_b64 = _tensor_to_b64(A)
b_b64 = _tensor_to_b64(b)

A_recv = _b64_to_tensor(A_b64)
b_recv = _b64_to_tensor(b_b64)

assert torch.allclose(A, A_recv), "A mismatch"
assert torch.allclose(b, b_recv), "b mismatch"
print("Storage encode/decode OK")
