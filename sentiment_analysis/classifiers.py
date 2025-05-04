from transformers import AutoModel
from torch import nn

from typing import Iterable


SENTIMENTS_AS_INDEX = {
    'positive': 0,
    'neutral': 1,
    'negative': 2
}

class BaseClassifierWithPhoBERT(nn.Module):
    """
    Base class for classification model with PhoBERT. The whole PhoBERT model is frozen
        and will not be trained during backpropogation.

    This Module will receive a batch of input_ids and attention mask output from the PhoBERT tokenizer.
    Then, both are passed into the PhoBERT model to create embeddings, from which the output corresponding
    to the first token is fed into the Module's `hidden_layer`.
    """
    def __init__(self, hidden_layer: nn.Module):
        """
        Initialize a `BaseClassifierWithPhoBERT`.

        :param torch.nn.Module hidden_layer: The hidden layer to perform
            classification on the PhoBERT's embeddings.
        """
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        # Freeze everything
        self.phobert.requires_grad_(False)
        
        self.hidden_layer = hidden_layer

    def forward(self, text_input_ids, attention_mask=None):
        phobert_embeddings = self.phobert(text_input_ids, attention_mask)
        return self.hidden_layer(phobert_embeddings['last_hidden_state'][:, 0, :])
    
class MLPClassifierWithPhoBERT(BaseClassifierWithPhoBERT):
    """
    Multi-layer Perceptron model with PhoBERT. The whole PhoBERT model is frozen
        and will not be trained during backpropogation.

    This Module will receive a batch of input_ids and attention mask output from the PhoBERT tokenizer.
    Then, both are passed into the PhoBERT model to create embeddings, from which the output corresponding
        to the first token is fed into the Module's `hidden_layer`, which comprises of a variable number of 
        fully connected layers and activation function.
    """
    def __init__(self,
        inner_dims: Iterable[int] | None = None,
        activation_fn: None | nn.Module = None
    ):
        """
        Initialize a `MLPClassifierWithPhoBERT`.

        :param Iterable[int] | None inner_dims: The dimension of the fully connected layer.
            For example: `[256]` will result in one fully connected layer with a dimension of 256.
            If `None` is given, the embeddings will be connected straight to the output layer.s
        :param nn.Module | None: The activation function betwen each layer. If `None`, the ReLU
            funtion is used.
        """
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