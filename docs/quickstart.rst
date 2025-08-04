Quick Start Guide
================

This guide will get you up and running with Photonic Flash Attention in minutes.

Installation
------------

Install from PyPI (recommended):

.. code-block:: bash

   pip install photonic-flash-attention

For hardware support:

.. code-block:: bash

   pip install photonic-flash-attention[hardware]

For development:

.. code-block:: bash

   git clone https://github.com/danieleschmidt/photonic-mlir-synth-bridge.git
   cd photonic-mlir-synth-bridge
   pip install -e ".[dev,benchmark]"

Basic Usage
-----------

Replace Standard Attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replace your existing attention layer:

.. code-block:: python

   # Before: Standard PyTorch attention
   attention = nn.MultiHeadAttention(embed_dim=768, num_heads=12)

   # After: Photonic Flash Attention
   from photonic_flash_attention import PhotonicFlashAttention
   attention = PhotonicFlashAttention(embed_dim=768, num_heads=12)

The API is fully compatible with PyTorch's MultiHeadAttention.

Drop-in Transformer Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from transformers import AutoModel
   from photonic_flash_attention import convert_to_photonic

   # Load existing model
   model = AutoModel.from_pretrained('bert-base-uncased')

   # Convert to use photonic attention
   photonic_model = convert_to_photonic(model)

   # Use normally - photonic acceleration is transparent
   inputs = tokenizer("Hello world!", return_tensors="pt")
   outputs = photonic_model(**inputs)

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

Configure for your specific use case:

.. code-block:: python

   from photonic_flash_attention import PhotonicFlashAttention

   attention = PhotonicFlashAttention(
       embed_dim=1024,
       num_heads=16,
       dropout=0.1,
       photonic_threshold=256,  # Use photonics for seq_len > 256
       device='auto',           # Auto-detect hardware
       enable_scaling=True,     # Enable concurrent processing
   )

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

Monitor performance in real-time:

.. code-block:: python

   # Perform attention computation
   output = attention(query, key, value)

   # Get performance statistics
   stats = attention.get_performance_stats()
   print(f"Device used: {stats['last_device_used']}")
   print(f"Latency: {stats['last_latency_ms']:.2f}ms")
   print(f"Energy: {stats['last_energy_mj']:.2f}mJ")
   print(f"Photonic usage: {stats['photonic_usage_ratio']:.1%}")

Configuration Options
--------------------

Global Configuration
~~~~~~~~~~~~~~~~~~~~

Set global options for all attention modules:

.. code-block:: python

   from photonic_flash_attention import set_global_config

   set_global_config(
       photonic_threshold=512,
       auto_device_selection=True,
       enable_profiling=True,
       log_level="INFO"
   )

Environment Variables
~~~~~~~~~~~~~~~~~~~~

Configure via environment variables:

.. code-block:: bash

   export PHOTONIC_THRESHOLD=512
   export PHOTONIC_LOG_LEVEL=INFO
   export PHOTONIC_SIMULATION=false  # Use real hardware
   export CUDA_VISIBLE_DEVICES=0

Device Information
------------------

Check available devices:

.. code-block:: python

   from photonic_flash_attention import get_device_info

   info = get_device_info()
   print(f"CUDA available: {info['cuda_available']}")
   print(f"Photonic available: {info['photonic_available']}")
   print(f"GPU devices: {info['cuda_device_count']}")

Example Workflows
-----------------

Training a Model
~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from photonic_flash_attention import PhotonicFlashAttention

   class TransformerBlock(nn.Module):
       def __init__(self, embed_dim, num_heads):
           super().__init__()
           self.attention = PhotonicFlashAttention(embed_dim, num_heads)
           self.norm1 = nn.LayerNorm(embed_dim)
           self.norm2 = nn.LayerNorm(embed_dim)
           self.feed_forward = nn.Sequential(
               nn.Linear(embed_dim, 4 * embed_dim),
               nn.GELU(),
               nn.Linear(4 * embed_dim, embed_dim),
           )

       def forward(self, x):
           # Self-attention with residual connection
           attn_out = self.attention(x)
           x = self.norm1(x + attn_out)
           
           # Feed-forward with residual connection
           ff_out = self.feed_forward(x)
           x = self.norm2(x + ff_out)
           
           return x

   # Training loop
   model = TransformerBlock(embed_dim=768, num_heads=12)
   optimizer = torch.optim.AdamW(model.parameters())

   for batch in dataloader:
       optimizer.zero_grad()
       output = model(batch)
       loss = criterion(output, targets)
       loss.backward()
       optimizer.step()

Inference with Batching
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from photonic_flash_attention import PhotonicFlashAttention

   # Create attention module optimized for inference
   attention = PhotonicFlashAttention(
       embed_dim=768,
       num_heads=12,
       dropout=0.0,  # Disable dropout for inference
       photonic_threshold=128,  # Lower threshold for inference
   )

   # Set to evaluation mode
   attention.eval()

   # Process multiple sequences
   sequences = [
       torch.randn(1, 256, 768),   # Short sequence
       torch.randn(1, 1024, 768),  # Medium sequence  
       torch.randn(1, 4096, 768),  # Long sequence
   ]

   with torch.no_grad():
       for i, seq in enumerate(sequences):
           output = attention(seq)
           stats = attention.get_performance_stats()
           print(f"Sequence {i}: {stats['last_device_used']} "
                 f"({stats['last_latency_ms']:.2f}ms)")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Issue**: No photonic hardware detected
**Solution**: Enable simulation mode for development:

.. code-block:: bash

   export PHOTONIC_SIMULATION=1

**Issue**: CUDA out of memory
**Solution**: Reduce batch size or sequence length:

.. code-block:: python

   # Process in smaller chunks
   def process_long_sequence(attention, sequence, chunk_size=1024):
       outputs = []
       for i in range(0, sequence.size(1), chunk_size):
           chunk = sequence[:, i:i+chunk_size, :]
           output = attention(chunk)
           outputs.append(output)
       return torch.cat(outputs, dim=1)

**Issue**: Poor performance on short sequences
**Solution**: Adjust photonic threshold:

.. code-block:: python

   attention.set_photonic_threshold(1024)  # Only use photonics for longer sequences

Debug Mode
~~~~~~~~~~

Enable detailed logging:

.. code-block:: python

   import logging
   logging.getLogger('photonic_flash_attention').setLevel(logging.DEBUG)

Or via environment:

.. code-block:: bash

   export PHOTONIC_LOG_LEVEL=DEBUG

Performance Profiling
~~~~~~~~~~~~~~~~~~~~~

Enable performance profiling:

.. code-block:: python

   from photonic_flash_attention import set_global_config

   set_global_config(enable_profiling=True)

   # Run your model
   output = attention(query)

   # Get detailed profiling info
   stats = attention.get_performance_stats()
   print(f"Profiling data: {stats}")

Next Steps
----------

* Read the :doc:`tutorials/index` for detailed examples
* Check out :doc:`api_reference` for complete API documentation
* See :doc:`hardware` for hardware setup and configuration
* Visit :doc:`performance` for optimization tips
* Join our community and :doc:`contributing` to the project

That's it! You're now ready to accelerate your attention computations with photonic hardware. 🚀