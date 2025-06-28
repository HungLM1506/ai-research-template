import torch 
import torch.nn as nn
import numpy as np 

from abc import abstractmethod



class BaseModel(nn.Module):
    """Base class for all models"""
    
    def __init__(self,config: dict[str, int | str | float | bool]) -> None:
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError("Forward method must be implemented in the subclass")
    
    def __str__(self) -> str:
        """Model print with number of traineable parameters"""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return f"{super().__str__()}\nTrainable parameters: {params}"