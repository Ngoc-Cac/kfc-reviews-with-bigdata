from transformers import AutoModel
from torch import nn


class ReviewClassifierWithPhoBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        # Freeze everything except for the RoBERTa pooling layer
        self.phobert.requires_grad_(False)
        self.phobert.pooler.requires_grad_(True)
        
        self.mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, text_input_ids, attention_mask):
        phobert_embeddings = self.phobert(text_input_ids, attention_mask)
        return self.mlp(phobert_embeddings['pooler_output'])