from pydantic import BaseModel
from typing import NamedTuple


class Size(NamedTuple):
    x: int
    y: int


class TrainModel(BaseModel):
    epoch: int
    output_size: Size
    batch_size: int
    version: int
