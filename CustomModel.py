"""
Custom Transformer Encoder Model for Sentiment Classification.
This module defines a custom Transformer Encoder architecture, designed to operate on top of
pre-trained BERT output embeddings. It includes a feed-forward layer, a multi-head self-attention
encoder block, and a stackable encoder model for classification tasks.
Classes:
    FeedForwardLayer(nn.Module):
        Implements a position-wise feed-forward layer with residual connection, dropout, and layer normalization.
            d_model (int): Dimensionality of the input and output features.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
        Methods:
            forward(x):
                Forward pass through the feed-forward layer.
    EncoderBlock(nn.Module):
        Implements a single Transformer encoder block with multi-head self-attention and a feed-forward layer.
            d_model (int): Dimensionality of the input and output features.
            num_heads (int): Number of attention heads.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
            mask (torch.Tensor, optional): Optional attention mask.
        Methods:
            forward(x):
                Forward pass through the encoder block.
    TransformerEncoder(nn.Module):
        Stacks multiple EncoderBlocks and applies a classification head.
            d_model (int): Dimensionality of the input and output features.
            num_heads (int): Number of attention heads per encoder block.
            num_layers (int): Number of encoder blocks to stack.
            num_classes (int): Number of output classes for classification.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
        Methods:
            forward(bert_output_embeddings):
                Forward pass through the stacked encoder blocks and classification head.
Example:
    >>> model = TransformerEncoder(
    ...     d_model=768,
    ...     num_heads=4,
    ...     num_layers=2,
    ...     num_classes=1,
    ...     dropout_rate=0.1
    ... )
    >>> bert_output = torch.randn(8, 128, 768)  # batch_size=8, seq_len=128, d_model=768
    >>> logits = model(bert_output)
"""
import math
from torch import nn
from torch.nn import Softmax


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(d_model, 2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, d_model)
        self.dropout = nn.Dropout(dropout_rate)  # Added dropout
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)  # Apply dropout after fc2
        x = self.layerNorm(x + residual)
        return x


class EncoderBlock(nn.Module):  # Renamed to EncoderBlock to signify it's one block
    def __init__(self, d_model, num_heads, dropout_rate=0.1, mask=None):
        super(EncoderBlock, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)
        # Dropout for attention output
        self.dropout_attn = nn.Dropout(dropout_rate)
        self.layerNorm_attn = nn.LayerNorm(
            d_model)  # LayerNorm for attention part

        # Pass dropout rate to FFL
        self.ffl = FeedForwardLayer(d_model, dropout_rate)

        self.mask = mask  # Still here for potential future use

    def split_heads(self, x):
        """Split the input tensor into multiple heads."""
        batch_size, seq_length, d_model = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)
        # (batch_size, num_heads, seq_length, head_dim)
        return x.permute(0, 2, 1, 3)

    def dot_prod(self, queries, keys):
        return queries @ keys.transpose(-2, -1)

    def scaler(self, queries, keys):
        """Scaled dot product attention with masking."""
        d_k = queries.size(-1)  # This is head_dim
        scores = self.dot_prod(queries, keys) / math.sqrt(d_k)
        if self.mask is not None:
            # Expand mask to match attention scores shape (batch, heads, seq_len, seq_len)
            scores = scores.masked_fill(self.mask == 0, float('-inf'))
        return Softmax(dim=-1)(scores)  # Or F.softmax(scores, dim=-1)

    def calc_attention(self, queries, keys, values):
        """Calculate the weighted sum of values based on attention scores."""
        attention_weights = self.scaler(queries, keys)
        attention = attention_weights @ values
        return attention

    def forward(self, x):
        # --- Multi-Head Self-Attention ---
        residual_attn = x
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)

        attention_output = self.calc_attention(queries, keys, values)

        # Concatenate heads
        batch_size, num_heads, seq_length, head_dim = attention_output.size()
        attention_output = attention_output.permute(
            0, 2, 1, 3).contiguous()  # (B, S, H, D/H)
        attention_output = attention_output.view(
            batch_size, seq_length, num_heads * head_dim)  # (B, S, D)

        # Final linear projection after attention
        attention_output = self.out(attention_output)

        # Apply dropout to attention output
        attention_output = self.dropout_attn(attention_output)

        # Add residual connection and Layer Normalization
        x = self.layerNorm_attn(residual_attn + attention_output)

        # --- Feed-Forward Layer ---
        # FFL handles its own residual and LayerNorm internally
        x = self.ffl(x)

        return x

# New wrapper for the full Encoder model (stack of EncoderBlocks)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, num_classes, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model

        # IMPORTANT: Positional Encoding is OMITTED here because it's assumed
        # the input `bert_output_embeddings` already contains positional information
        # from a pre-trained BERT model.

        # Stack of Encoder Blocks
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])

        # Final Classification Head
        self.outfinal = nn.Linear(d_model, num_classes)

    def forward(self, bert_output_embeddings):
        """
        Args:
            bert_output_embeddings (torch.Tensor): The output tensor from a BERT model's
                                                   last hidden state.
                                                   Shape: (batch_size, sequence_length, d_model)
        """
        x = bert_output_embeddings

        # Pass through all encoder blocks
        for layer in self.layers:
            x = layer(x)

        # Pool the sequence output for classification
        pooled = x.mean(dim=1)  # (batch_size, d_model)

        # Classification
        logits = self.outfinal(pooled)
        return logits
