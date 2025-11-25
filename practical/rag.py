import torch
from typing import Callable, List
class RaggedTensor:
    """
    A PyTorch implementation of a RaggedTensor-like structure.

    Attributes:
        flat_values (torch.Tensor): The flat values of the ragged tensor.
        row_splits (torch.Tensor): The row splits defining the ragged structure.
    """

    def __init__(self, flat_values: torch.Tensor, row_splits: torch.Tensor = None):
        """
        Initializes the RaggedTensor.

        Args:
            flat_values (torch.Tensor): The flat values of the ragged tensor.
            row_splits (torch.Tensor): The row splits defining the ragged structure.
        """
        

        assert row_splits.ndim == 1, "row_splits must be a 1D tensor."
        assert row_splits[0] == 0, "row_splits must start with 0."
        assert torch.all(row_splits[1:] >= row_splits[:-1]), "row_splits must be non-decreasing."
        assert row_splits[-1] == flat_values.size(0), "Last value of row_splits must equal flat_values size."

        self.flat_values = flat_values
        self.row_splits = row_splits

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
    def from_nested_list(cls, nested_list: list, dtype=torch.float32):
        """
        Creates a RaggedTensor from a nested list.

        Args:
            nested_list (list): A nested list representing the ragged structure.
            dtype (torch.dtype): The data type of the tensor.

        Returns:
            RaggedTensor: A new RaggedTensor instance.
        """
        flat_values = [item for sublist in nested_list for item in sublist]  # Flatten the list
        row_splits = [0]
        for sublist in nested_list:
            row_splits.append(row_splits[-1] + len(sublist))

        return cls(flat_values=torch.tensor(flat_values, dtype=dtype),
                   row_splits=torch.tensor(row_splits, dtype=torch.int64))

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

    def reduce_prod(self, dim = 1, keepdim = False):
        # Initialize an empty list to store the row-wise products
        row_products = []

        # Iterate through the row_splits to compute row products
        for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
            row = self.flat_values[start:end]  # Extract the row
            if row.numel() > 0:
                row_products.append(row.prod(dim-1, keepdim))  # Compute the product of the row
            else:
                row_products.append(torch.tensor(1.0))  # Handle empty rows (product is 1)

        # Convert the list of row products to a tensor
        return torch.stack(row_products)
        

    def reduce_min(self, dim = 1, keepdim = False):
        # Initialize an empty list to store the row-wise mins
        row_min = []

        # Iterate through the row_splits to compute row mins
        for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
            row = self.flat_values[start:end]  # Extract the row
            if row.numel() > 0:
                row_min.append(row.min(dim-1, keepdim).values)  # Compute the min of the row
            else:
                row_min.append(torch.tensor(1.0))  # Handle empty rows (min is 1)
        return torch.stack(row_min)

    def reduce_sum(self, dim = 1, keepdim = False):
        row_sum = []

        # Iterate through the row_splits to compute row sums
        for start, end in zip(self.row_splits[:-1], self.row_splits[1:]):
            row = self.flat_values[start:end]  # Extract the row
            if row.numel() > 0:
                row_sum.append(row.sum(dim-1, keepdim))  # Compute the sum of the row
            else:
                row_sum.append(torch.tensor(1.0))  # Handle empty rows (sum is 1)
        # first_tensor_shape = row_sum[0].shape
        # if all(tensor.shape == first_tensor_shape for tensor in row_sum):
        return torch.stack(row_sum)
        


    def value_rowids(self):
        row_ids = torch.cat([torch.full((len(row),), i, dtype=torch.long) for i, row in enumerate(self.to_list())])
        # values = torch.cat(self.to_list())
        return row_ids
    
    def sign(self):
        sign = torch.sign(self.flat_values)
        return RaggedTensor(flat_values=sign, row_splits=self.row_splits)
    
    def __repr__(self):
        return f"RaggedTensor({self.to_list()})"

