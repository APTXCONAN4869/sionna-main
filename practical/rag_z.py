import torch
from typing import Callable, List
class RaggedTensor:
    def __init__(self, flat_values: torch.Tensor, row_splits: torch.Tensor):
        assert row_splits.ndim == 1
        assert row_splits[0] == 0
        assert torch.all(row_splits[1:] >= row_splits[:-1])
        assert row_splits[-1] == flat_values.size(0)

        self.flat_values = flat_values
        self.row_splits = row_splits

    def to(self, device):
        """Move both flat_values and row_splits to a given device."""
        self.flat_values = self.flat_values.to(device)
        self.row_splits = self.row_splits.to(device)
        return self

    @classmethod
    def from_row_splits(cls, values: torch.Tensor, row_splits: torch.Tensor):
        """
        Creates a RaggedTensor from flat values and row splits.

        Args:
            values (torch.Tensor): Flat values of the tensor.
            row_splits (torch.Tensor): Row splits defining the ragged structure.

        Returns:
            RaggedTensor: A new instance of RaggedTensor.
        """
        return cls(flat_values=values, row_splits=row_splits)

    def with_flat_values(self, new_flat_values: torch.Tensor):
        """
        Creates a new RaggedTensor with the same row_splits but new flat_values.

        Args:
            new_flat_values (torch.Tensor): New flat values for the ragged tensor.

        Returns:
            RaggedTensor: A new instance of RaggedTensor with updated flat values.
        """
        # Ensure the new flat_values size matches the original size
        assert new_flat_values.size(0) == self.flat_values.size(0), \
            "new_flat_values must have the same size as the original flat_values."

        return RaggedTensor(flat_values=new_flat_values, row_splits=self.row_splits)

    def map_flat_values(self, func: Callable, *args, **kwargs) -> "RaggedTensor":
        """
        Applies a function to transform the flat values, or generates new flat values
        entirely if self.flat_values are not needed.

        Args:
            func (Callable): A function to generate new flat values. It can ignore self.flat_values.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            RaggedTensor: A new RaggedTensor with transformed or replaced flat values.
        """
        # Call the function with or without self.flat_values
        new_flat_values = func(self.flat_values, *args, **kwargs) if self.flat_values is not None \
                        else func(*args, **kwargs)
        return RaggedTensor(new_flat_values, self.row_splits)

    def to_list(self) -> List[List]:
        """
        Converts the ragged tensor to a nested Python list.

        Returns:
            List[List]: A nested list representation of the ragged tensor.
        """
        result = []
        for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
            result.append(torch.tensor(self.flat_values[start:end].tolist()))
        return result

    @classmethod
    def from_nested_list(cls, nested_list: list, dtype=torch.float32, device=None):
        flat_values = [item for sublist in nested_list for item in sublist]
        row_splits = [0]
        for sublist in nested_list:
            row_splits.append(row_splits[-1] + len(sublist))

        flat = torch.tensor(flat_values, dtype=dtype, device=device)
        splits = torch.tensor(row_splits, dtype=torch.int64, device=device)
        return cls(flat, splits)

    def expand_dims(self, dim=-1):
        """
        Expands the dimensions of the flat_values tensor.

        Args:
            axis (int): The dimension index where the new axis is added.

        Returns:
            RaggedTensor: A new RaggedTensor with expanded flat_values.
        """
        new_flat_values = self.flat_values.unsqueeze(dim)  # Add dimension
        return RaggedTensor(flat_values=new_flat_values, row_splits=self.row_splits)



        return result

    def reduce_prod(self, keepdim=False):
        """Fully GPU-accelerated product per row."""
        device = self.flat_values.device
        num_rows = len(self.row_splits) - 1
        feature_dim = self.flat_values.size(1)
        result = torch.ones((num_rows, feature_dim), device=device, dtype=self.flat_values.dtype)
        for i in range(num_rows):
            start, end = self.row_splits[i].item(), self.row_splits[i + 1].item()
            if start < end:
                result[i] = self.flat_values[start:end].prod()
        if keepdim:
            result = result.unsqueeze(-1)
        return result

    def reduce_min(self, dim = 1, keepdim = False):
        # Initialize an empty list to store the row-wise mins
        device = self.flat_values.device
        num_rows = len(self.row_splits) - 1
        feature_dim = self.flat_values.size(1)
        result = torch.ones((num_rows, feature_dim), device=device, dtype=self.flat_values.dtype)
        for i in range(num_rows):
            start, end = self.row_splits[i].item(), self.row_splits[i + 1].item()
            if start < end:
                result[i] = self.flat_values[start:end].min()
        if keepdim:
            result = result.unsqueeze(-1)
        return result

    def reduce_sum(self, keepdim=False):
        """Fully GPU-accelerated version of reduce_sum."""
        device = self.flat_values.device
        row_ids = torch.repeat_interleave(
            torch.arange(len(self.row_splits) - 1, device=device),
            self.row_splits[1:] - self.row_splits[:-1]
        )
        # result may not be 1D tensor if flat_values is not 1D
        num_rows = len(self.row_splits) - 1
        feature_dim = self.flat_values.size(1)
        result = torch.zeros((num_rows, feature_dim), device=device, dtype=self.flat_values.dtype)
        result.index_add_(0, row_ids, self.flat_values)
        if keepdim:
            result = result.unsqueeze(-1)
        return result

    def value_rowids(self):
        """GPU-friendly row index mapping."""
        device = self.flat_values.device
        row_ids = torch.repeat_interleave(
            torch.arange(len(self.row_splits) - 1, device=device),
            self.row_splits[1:] - self.row_splits[:-1]
        )
        return row_ids

    def sign(self):
        return RaggedTensor(torch.sign(self.flat_values), self.row_splits)

    def __repr__(self):
        return f"RaggedTensor(flat_values={self.flat_values}, row_splits={self.row_splits})"
