import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    char_to_idx = {c: i for i, c in enumerate(sorted(set(text)))}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    text_clean = ''.join(c for c in text if c not in chars_to_remove)
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    result = torch.zeros((len(text), len(char_to_idx)), dtype=torch.int8)
    for i, char in enumerate(text):
        result[i, char_to_idx[char]] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    result = ''.join(idx_to_char[i.item()] for i in embedded_text.argmax(dim=1))
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    n_sequences = (len(text) - 1) // seq_len
    x = chars_to_onehot(text[:n_sequences * seq_len], char_to_idx)
    y = torch.tensor([char_to_idx[c] for c in text[1:n_sequences * seq_len + 1]])
    return x.reshape(n_sequences, seq_len, -1).to(device), y.reshape(n_sequences, seq_len).to(device)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    return torch.softmax(y / max(temperature, 1e-8), dim=dim)
    # ========================


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    with torch.no_grad():
        h = None
        x = chars_to_onehot(start_sequence, char_to_idx).unsqueeze(0).to(device, dtype=torch.float)
        
        while len(out_text) < n_chars:
            y, h = model(x, h)
            probs = hot_softmax(y[0, 0], dim=0, temperature=T)
            next_char_idx = torch.multinomial(probs, 1).item()
            out_text += idx_to_char[next_char_idx]
            x = chars_to_onehot(idx_to_char[next_char_idx], char_to_idx).unsqueeze(0).to(device, dtype=torch.float)
    # ========================
    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        n_batches = len(self.dataset) // self.batch_size
        indices = torch.arange(n_batches * self.batch_size)
        indices = indices.view(self.batch_size, n_batches).t().reshape(-1)
        
        idx = indices.tolist()
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: READ THIS SECTION!!

        # ====== YOUR CODE: ======
        # Create parameter groups for each layer
        for i in range(n_layers):
            input_size = self.in_dim if i == 0 else self.h_dim
            
            layer_dict = {
                "update_x": nn.Linear(input_size, self.h_dim, bias=False),
                "update_h": nn.Linear(self.h_dim, self.h_dim, bias=True),
                "reset_x": nn.Linear(input_size, self.h_dim, bias=False),
                "reset_h": nn.Linear(self.h_dim, self.h_dim, bias=True),
                "candidate_x": nn.Linear(input_size, self.h_dim, bias=False),
                "candidate_h": nn.Linear(self.h_dim, self.h_dim, bias=True),
                "dropout": nn.Dropout(p=dropout)
            }
            
            self.layer_params.append(layer_dict)
            
            for key, module in layer_dict.items():
                self.add_module(f"{key}_layer{i}", module)
        
        self.output_layer = nn.Linear(self.h_dim, self.out_dim, bias=True)
        self.layer_params.append({"output": self.output_layer})
        self.add_module("output_layer", self.output_layer)

        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO: READ THIS SECTION!!
        # ====== YOUR CODE: ======
        layer_output = torch.zeros(batch_size, seq_len, self.out_dim, device=input.device)

        # Process each time step
        for t in range(seq_len):
            x_t = layer_input[:, t]
            
            # Process through each layer
            for layer_idx in range(self.n_layers):
                # Get current layer parameters and state
                h_t = layer_states[layer_idx]
                layer_dict = self.layer_params[layer_idx]
                
                # Update input for hidden layers
                if layer_idx > 0:
                    x_t = layer_states[layer_idx-1]
                
                # Compute gate values
                z = torch.sigmoid(
                    layer_dict["update_x"](x_t) + 
                    layer_dict["update_h"](h_t)
                )
                r = torch.sigmoid(
                    layer_dict["reset_x"](x_t) + 
                    layer_dict["reset_h"](h_t)
                )
                
                # Compute new hidden state candidate
                g = torch.tanh(
                    layer_dict["candidate_x"](x_t) + 
                    layer_dict["candidate_h"](r * h_t)
                )
                
                # Update hidden state
                new_h = z * h_t + (1 - z) * g
                layer_states[layer_idx] = new_h
                
                # Apply dropout between layers (not on the last layer)
                if layer_idx < self.n_layers - 1:
                    x_t = layer_dict["dropout"](new_h)
            
            # Compute output for current timestep
            layer_output[:, t] = self.layer_params[-1]["output"](layer_states[-1])
        
        # Stack final hidden states
        hidden_state = torch.stack(layer_states, dim=1)
        # ========================
        return layer_output, hidden_state