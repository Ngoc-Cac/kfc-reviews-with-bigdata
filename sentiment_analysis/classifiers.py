from transformers import AutoModel
from torch import nn


class BaseClassifierWithPhoBERT(nn.Module):
    def __init__(self, hidden_layer):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        # Freeze everything except for the RoBERTa pooling layer
        self.phobert.requires_grad_(False)
        self.phobert.pooler.requires_grad_(True)
        
        self.hidden_layer = hidden_layer

    def forward(self, text_input_ids, attention_mask):
        phobert_embeddings = self.phobert(text_input_ids, attention_mask)
        return self.hidden_layer(phobert_embeddings['pooler_output'])
    
class MLPClassifierWithPhoBERT(BaseClassifierWithPhoBERT):
    def __init__(self, inner_dim: int = 512, use_relu: bool = True):
        activation_fn = nn.ReLU() if use_relu else nn.Sigmoid()
        
        hidden_layer = nn.Sequential(
            nn.Linear(768, inner_dim),
            activation_fn,
            nn.Linear(inner_dim, inner_dim),
            activation_fn,
            nn.Linear(inner_dim, 3)
        )
        super().__init__(hidden_layer)