#!/usr/bin/env python3
"""
Transformer integration example for Photonic Flash Attention.

This example shows how to integrate photonic attention into:
1. Custom transformer models
2. Hugging Face transformers
3. Training and inference workflows
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

from photonic_flash_attention import PhotonicFlashAttention, convert_to_photonic


class PhotonicTransformerBlock(nn.Module):
    """A transformer block using photonic attention."""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = PhotonicFlashAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, attention_mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, attention_mask=attention_mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class PhotonicTransformer(nn.Module):
    """Complete transformer model using photonic attention."""
    
    def __init__(
        self, 
        vocab_size, 
        embed_dim, 
        num_heads, 
        num_layers, 
        max_seq_len, 
        num_classes=None,
        dropout=0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            PhotonicTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=4 * embed_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        if num_classes is not None:
            # Classification head
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            # Language modeling head
            self.lm_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        
        x = self.layer_norm(x)
        
        # Output projection
        if hasattr(self, 'classifier'):
            # Classification: use [CLS] token (first token)
            return self.classifier(x[:, 0, :])
        else:
            # Language modeling: predict next token for all positions
            return self.lm_head(x)
    
    def get_attention_stats(self):
        """Get attention performance statistics from all blocks."""
        stats = {}
        for i, block in enumerate(self.blocks):
            block_stats = block.attention.get_performance_stats()
            stats[f'block_{i}'] = block_stats
        return stats


def create_synthetic_data(vocab_size, seq_len, batch_size, num_batches):
    """Create synthetic data for demonstration."""
    data = []
    labels = []
    
    for _ in range(num_batches):
        # Random token sequences
        batch_data = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Random classification labels
        batch_labels = torch.randint(0, 2, (batch_size,))
        
        data.append(batch_data)
        labels.append(batch_labels)
    
    return torch.cat(data, dim=0), torch.cat(labels, dim=0)


def train_model_example():
    """Example of training a photonic transformer model."""
    print("🏋️ Training Photonic Transformer")
    print("=" * 40)
    
    # Model configuration
    config = {
        'vocab_size': 10000,
        'embed_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'max_seq_len': 512,
        'num_classes': 2,  # Binary classification
        'dropout': 0.1,
    }
    
    # Create model
    model = PhotonicTransformer(**config)
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create synthetic dataset
    seq_len = 256
    batch_size = 8
    num_batches = 10
    
    data, labels = create_synthetic_data(
        config['vocab_size'], seq_len, batch_size, num_batches
    )
    
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training loop
    model.train()
    epoch_loss = 0
    epoch_start = time.time()
    
    print(f"Training on {len(dataset)} samples...")
    
    for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Print progress
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.4f}")
    
    epoch_time = time.time() - epoch_start
    avg_loss = epoch_loss / len(dataloader)
    
    print(f"Training completed: Avg Loss = {avg_loss:.4f}, Time = {epoch_time:.2f}s")
    
    # Get attention statistics
    attention_stats = model.get_attention_stats()
    print("\nAttention Statistics:")
    for block_name, stats in attention_stats.items():
        device_used = stats.get('last_device_used', 'unknown')
        latency = stats.get('last_latency_ms', 0)
        print(f"  {block_name}: {device_used} ({latency:.2f}ms)")
    
    print()
    return model


def inference_example(model):
    """Example of inference with a trained model."""
    print("🔮 Inference Example")
    print("=" * 20)
    
    model.eval()
    
    # Create test data with different sequence lengths
    test_configs = [
        {'seq_len': 128, 'batch_size': 1},
        {'seq_len': 512, 'batch_size': 1},
        {'seq_len': 1024, 'batch_size': 1},
    ]
    
    print(f"{'Seq Len':>8} | {'Device':>10} | {'Latency (ms)':>12} | {'Prediction':>10}")
    print("-" * 50)
    
    with torch.no_grad():
        for config in test_configs:
            seq_len = config['seq_len']
            batch_size = config['batch_size']
            
            # Create test input
            test_input = torch.randint(0, 1000, (batch_size, seq_len))
            
            # Inference
            start_time = time.time()
            output = model(test_input)
            end_time = time.time()
            
            # Get prediction
            prediction = torch.argmax(output, dim=-1).item()
            latency_ms = (end_time - start_time) * 1000
            
            # Get device info from first attention block
            stats = model.blocks[0].attention.get_performance_stats()
            device_used = stats.get('last_device_used', 'unknown')
            
            print(f"{seq_len:>8} | {device_used:>10} | {latency_ms:>12.2f} | {prediction:>10}")
    
    print()


def huggingface_integration_example():
    """Example of integrating with Hugging Face transformers."""
    print("🤗 Hugging Face Integration")
    print("=" * 30)
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # Load a small model for demonstration
        model_name = "distilbert-base-uncased"
        print(f"Loading {model_name}...")
        
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Convert to use photonic attention
        print("Converting to photonic attention...")
        photonic_model = convert_to_photonic(
            model,
            photonic_config={
                'min_seq_length': 128,
                'max_seq_length': 512,
                'wavelengths': 64,
            }
        )
        
        # Test with sample text
        texts = [
            "Hello, world!",
            "This is a longer text that should trigger photonic computation when the sequence length exceeds the threshold.",
            "Short text.",
        ]
        
        print("\nProcessing sample texts:")
        print("-" * 40)
        
        for i, text in enumerate(texts):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            seq_len = inputs['input_ids'].shape[1]
            
            # Forward pass
            with torch.no_grad():
                outputs = photonic_model(**inputs)
            
            print(f"Text {i+1}: {seq_len} tokens, Output shape: {outputs.last_hidden_state.shape}")
        
        print("✅ Hugging Face integration successful!")
        
    except ImportError:
        print("⚠️  Hugging Face transformers not available - skipping this example")
        print("   Install with: pip install transformers")
    
    print()


def performance_comparison():
    """Compare performance between standard and photonic attention."""
    print("📊 Performance Comparison")
    print("=" * 30)
    
    # Configuration
    embed_dim = 768
    num_heads = 12
    batch_size = 4
    
    # Create standard and photonic attention modules
    standard_attention = nn.MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        batch_first=True,
    )
    
    photonic_attention = PhotonicFlashAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=0.0,
        photonic_threshold=256,
    )
    
    sequence_lengths = [128, 256, 512, 1024]
    
    print(f"{'Seq Len':>8} | {'Standard (ms)':>13} | {'Photonic (ms)':>13} | {'Speedup':>8}")
    print("-" * 55)
    
    for seq_len in sequence_lengths:
        # Create test data
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        
        # Warm-up
        _ = standard_attention(query, key, value)
        _ = photonic_attention(query, key, value)
        
        # Time standard attention
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        _ = standard_attention(query, key, value)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        standard_time = (time.perf_counter() - start) * 1000
        
        # Time photonic attention
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        _ = photonic_attention(query, key, value)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        photonic_time = (time.perf_counter() - start) * 1000
        
        speedup = standard_time / photonic_time
        
        print(f"{seq_len:>8} | {standard_time:>13.2f} | {photonic_time:>13.2f} | {speedup:>8.2f}x")
    
    print()


def main():
    """Main function to run all examples."""
    print("🌟 Photonic Flash Attention - Transformer Integration Examples")
    print("=" * 70)
    print()
    
    # Train a model
    model = train_model_example()
    
    # Run inference
    inference_example(model)
    
    # Hugging Face integration
    huggingface_integration_example()
    
    # Performance comparison
    performance_comparison()
    
    print("✅ All transformer integration examples completed successfully!")
    print()
    print("Next steps:")
    print("- Try with your own models and datasets")
    print("- Experiment with different photonic thresholds")
    print("- Monitor performance in production workloads")
    print("- Contribute improvements to the project!")


if __name__ == "__main__":
    main()