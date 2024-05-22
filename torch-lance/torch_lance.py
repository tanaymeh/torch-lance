import os
import lance
import pyarrow as pa
import torch

from collections import OrderedDict

GLOBAL_SCHEMA = pa.schema(
    [
        pa.field("name", pa.string()),
        pa.field("value", pa.list_(pa.float64(), -1)),
        pa.field("shape", pa.list_(pa.int64(), -1)),
    ]
)


def save_model(state_dict: OrderedDict, file_name: str, version=False):
    """Saves a PyTorch model in lance file format

    Args:
        state_dict (OrderedDict): Model state dict
        file_name (str): Lance model name
        version (bool): Whether to save as a new version or overwrite the existing versions,
            if the lance file already exists
    """
    # Create a reader
    reader = pa.RecordBatchReader.from_batches(
        GLOBAL_SCHEMA, _save_model_writer(state_dict)
    )

    if os.path.exists(file_name):
        if version:
            # If we want versioning, we use the overwrite mode to create a new version
            lance.write_dataset(
                reader, file_name, schema=GLOBAL_SCHEMA, mode="overwrite"
            )
        else:
            # If we don't want versioning, we delete the existing file and write a new one
            os.remove(file_name)
            lance.write_dataset(reader, file_name, schema=GLOBAL_SCHEMA)
    else:
        # If the file doesn't exist, we write a new one
        lance.write_dataset(reader, file_name, schema=GLOBAL_SCHEMA)


def load_model(
    model: torch.nn.Module, file_name: str, version: int = 1, map_location=None
):
    """Loads the model weights from lance file and sets them to the model

    Args:
        model (torch.nn.Module): PyTorch model
        file_name (str): Lance model name
        version (int): Version of the model to load
        map_location (str): Device to load the model on
    """
    state_dict = load_state_dict(file_name, version=version, map_location=map_location)
    model.load_state_dict(state_dict)


def load_state_dict(file_name: str, version: int = 1, map_location=None) -> OrderedDict:
    """Reads the model weights from lance file and returns a model state dict
    If the model weights are too large, this function will fail with a memory error.

    Args:
        file_name (str): Lance model name
        version (int): Version of the model to load
        map_location (str): Device to load the model on

    Returns:
        OrderedDict: Model state dict
    """
    ds = lance.dataset(file_name, version=version)
    weights = ds.take([x for x in range(ds.count_rows())]).to_pylist()
    state_dict = OrderedDict()

    for weight in weights:
        state_dict[weight["name"]] = _load_weight(weight).to(map_location)

    return state_dict


def _load_weight(weight: list) -> torch.Tensor:
    """Converts a weight list to a torch tensor"""
    return torch.tensor(weight["value"], dtype=torch.float64).reshape(weight["shape"])


def _save_model_writer(state_dict):
    """Yields a RecordBatch for each parameter in the model state dict"""
    for param_name, param in state_dict.items():
        param_shape = list(param.size())
        param_value = param.flatten().tolist()
        yield pa.RecordBatch.from_arrays(
            [
                pa.array(
                    [param_name],
                    pa.string(),
                ),
                pa.array(
                    [param_value],
                    pa.list_(pa.float64(), -1),
                ),
                pa.array(
                    [param_shape],
                    pa.list_(pa.int64(), -1),
                ),
            ],
            ["name", "value", "shape"],
        )
