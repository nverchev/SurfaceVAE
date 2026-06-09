"""Custom dataloaders and collate functions for model training."""

import torch
from typing_extensions import override

from drytorch import DataLoader
from src.data.structures import Input, Target
from src.utils.sparse import torch_sparse_block_diag


from typing import TypeGuard


def is_tensor_list(val: list[torch.Tensor | None]) -> TypeGuard[list[torch.Tensor]]:
    return all(x is not None for x in val)


def collate_coma_batch(batch: list[tuple[Input, Target]]) -> tuple[Input, Target]:
    """Collate a batch of COMA data by stacking shapes and block-diagonalizing operators."""
    xs = [item[0].x for item in batch]
    x = torch.stack(xs, 0)
    operators = [item[0].operator for item in batch]
    if not is_tensor_list(operators):
        operator = torch.empty(0)
        operator_adjoint = torch.empty(0)
    else:
        if len(operators) > 0 and all(op is operators[0] for op in operators):
            operator = operators[0]
        else:
            operator = torch_sparse_block_diag(operators)

        operators_adjoint = [item[0].operator_adjoint for item in batch]
        if len(operators_adjoint) > 0 and all(
            op is operators_adjoint[0] for op in operators_adjoint
        ):
            operator_adjoint = (
                operators_adjoint[0]
                if operators_adjoint[0] is not None
                else torch.empty(0)
            )
        else:
            assert is_tensor_list(operators_adjoint)
            operator_adjoint = torch_sparse_block_diag(operators_adjoint)

    inputs = Input(x=x, operator=operator, operator_adjoint=operator_adjoint)
    targets = Target(x=torch.stack([item[1].x for item in batch], 0))
    return inputs, targets


class COMADataLoader(DataLoader[tuple[Input, Target]]):
    """Custom DataLoader for COMA dataset splits."""

    @override
    def get_loader(self, inference: bool) -> torch.utils.data.DataLoader:
        loader = super().get_loader(inference)
        loader.collate_fn = collate_coma_batch
        return loader
