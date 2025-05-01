from transformers import AutoModel
from torch import nn

from typing import Iterable


class BaseClassifierWithPhoBERT(nn.Module):
    def __init__(self, hidden_layer):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        # Freeze everything
        self.phobert.requires_grad_(False)
        
        self.hidden_layer = hidden_layer

    def forward(self, text_input_ids, attention_mask=None):
        phobert_embeddings = self.phobert(text_input_ids, attention_mask)
        return self.hidden_layer(phobert_embeddings['last_hidden_state'][:, 0, :])
    
class MLPClassifierWithPhoBERT(BaseClassifierWithPhoBERT):
    def __init__(self,
        inner_dims: Iterable[int] | None = None,
        activation_fn: None | nn.Module = None
    ):
        if inner_dims is None:
            inner_dims = []
        if activation_fn is None:
            activation_fn = nn.ReLU()
        
        hidden_layer = nn.Sequential()
        prev_dim = 768
        for dim in inner_dims:
            hidden_layer.append(nn.Linear(prev_dim, dim))
            hidden_layer.append(activation_fn)

            prev_dim = dim

        hidden_layer.append(nn.Linear(prev_dim, 3))

        super().__init__(hidden_layer)