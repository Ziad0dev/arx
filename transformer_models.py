"""
Transformer-based models for the AI Research System
"""

from advanced_ai_analyzer import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for transformer models"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.shape[0]
        
        # Project inputs to multi-heads
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_proj(attn_output)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with multi-head attention and feed-forward network"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention block with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed forward block with residual connection and layer norm
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerClassifier(nn.Module):
    """Transformer-based classifier for research paper analysis"""
    
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = self._create_positional_encoding(embed_dim, 1)  # Only one position for paper embeddings
        self.transformer = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, embed_dim, max_len):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, embed_dim]
        
    def forward(self, x):
        # Project input to embedding dimension
        x = self.input_proj(x).unsqueeze(1)  # Add sequence dimension [batch_size, 1, embed_dim]
        
        # Add positional encoding
        x = x + self.positional_encoding.to(x.device)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global pooling (not needed for single position)
        x = x.squeeze(1)
        
        # Classification
        x = self.classifier(self.dropout(x))
        
        return x


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for knowledge graph reasoning"""
    
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Trainable parameters
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        
    def forward(self, h, adj):
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # [N, out_features]
        
        # Self-attention on the nodes
        N = Wh.size(0)
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        # Mask attention scores with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime


class KnowledgeGraphNetwork(nn.Module):
    """Graph Neural Network for knowledge graph reasoning"""
    
    def __init__(self, num_concepts, embedding_dim, hidden_dim, num_classes, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Concept embeddings
        self.concept_embeddings = nn.Embedding(num_concepts, embedding_dim)
        
        # Multiple graph attention layers with different heads (akin to multi-head attention)
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(embedding_dim, hidden_dim, dropout) for _ in range(num_heads)
        ])
        
        # Final projection
        self.out_proj = nn.Linear(hidden_dim * num_heads, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, indices, adj_matrix):
        # Get concept embeddings
        x = self.concept_embeddings(indices)
        
        # Apply multiple GAT layers
        outputs = [layer(x, adj_matrix) for layer in self.gat_layers]
        x = torch.cat(outputs, dim=1)
        
        # Final classification
        x = self.dropout(x)
        logits = self.out_proj(x)
        
        return logits


class ResearchPaperModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Split model across GPU if needed
        self.section_encoders = nn.ModuleList([
            AutoModel.from_pretrained(CONFIG['model_name']).to(f'cuda:{i%torch.cuda.device_count()}')
            for i in range(4)  # Encode abstract, intro, methods, results separately
        ])
        
    def forward(self, paper):
        """Process paper sections in parallel"""
        # Split paper into sections
        sections = self._split_into_sections(paper)
        
        # Encode sections in parallel on different GPUs
        section_embeddings = []
        for i, (encoder, section) in enumerate(zip(self.section_encoders, sections)):
            with torch.cuda.device(f'cuda:{i%torch.cuda.device_count()}'):
                section_embeddings.append(encoder(**section).last_hidden_state.mean(1))
                
        # Combine section embeddings
        return torch.mean(torch.stack(section_embeddings), dim=0)


def create_transformer_model(input_dim, output_dim, config=CONFIG):
    """Create a transformer model based on configuration"""
    embed_dim = config.get('transformer_embed_dim', 256)
    num_heads = config.get('transformer_num_heads', 4)
    ff_dim = config.get('transformer_ff_dim', 512)
    num_layers = config.get('transformer_num_layers', 2)
    dropout = config.get('dropout_rate', 0.1)
    
    return TransformerClassifier(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        num_classes=output_dim,
        dropout=dropout
    )


def create_knowledge_graph_model(num_concepts, output_dim, config=CONFIG):
    """Create a knowledge graph neural network model"""
    embedding_dim = config.get('kg_embedding_dim', 128)
    hidden_dim = config.get('kg_hidden_dim', 64)
    num_heads = config.get('kg_num_heads', 4)
    dropout = config.get('dropout_rate', 0.1)
    
    return KnowledgeGraphNetwork(
        num_concepts=num_concepts,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=output_dim,
        num_heads=num_heads,
        dropout=dropout
    )
