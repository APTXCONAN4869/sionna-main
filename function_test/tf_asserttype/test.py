import torch

def assert_type(tensor, expected_dtype):
    assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, but got {tensor.dtype}"

x = torch.randn(3, 4, dtype=torch.float32)
expected_dtype = torch.float32
assert_type(x, expected_dtype)  # pass
expected_dtype = torch.float16
assert_type(x, expected_dtype)  # error
