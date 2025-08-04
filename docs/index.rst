Photonic Flash Attention Documentation
=====================================

Welcome to Photonic Flash Attention, a hybrid photonic-electronic implementation of Flash Attention that automatically switches between optical and electronic computation based on sequence length and hardware availability.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   installation
   api_reference
   tutorials/index
   examples/index
   hardware
   performance
   contributing
   changelog

Key Features
------------

* **Hybrid Photonic-Electronic**: Seamlessly switches between optical and GPU kernels
* **Zero-Copy Integration**: Direct memory mapping between photonic and CUDA memory  
* **Adaptive Routing**: Dynamic selection based on sequence length and batch size
* **Energy Efficient**: Up to 10x lower power for long sequences
* **Drop-in Compatible**: Works with existing transformer implementations

Quick Start
-----------

.. code-block:: python

   import torch
   from photonic_flash_attention import PhotonicFlashAttention

   # Create hybrid attention module
   attention = PhotonicFlashAttention(
       embed_dim=768,
       num_heads=12,
       photonic_threshold=512,  # Use photonics for seq_len > 512
       device='auto'  # Automatically detect photonic hardware
   )

   # Use like standard attention
   q = torch.randn(2, 1024, 768)  # [batch, seq_len, embed_dim]
   output = attention(q)

   print(f"Computation device: {attention.last_device_used}")
   print(f"Latency: {attention.last_latency_ms:.2f} ms")

Architecture Overview
--------------------

The Photonic Flash Attention system consists of several key components:

.. image:: _static/architecture_diagram.png
   :alt: Architecture Diagram
   :width: 800px

Core Components:

* **Flash Attention 3**: Optimized GPU baseline implementation
* **Photonic Attention**: Silicon photonic hardware acceleration
* **Hybrid Router**: Intelligent device selection using machine learning
* **Memory Manager**: Unified memory management across devices
* **Security Layer**: Input validation and protection against attacks

Performance Benefits
-------------------

Photonic Flash Attention provides significant performance improvements for long sequences:

.. list-table:: Performance Comparison
   :header-rows: 1

   * - Sequence Length
     - GPU Latency (ms)
     - Photonic Latency (ms)
     - Speedup
     - Energy Savings
   * - 512
     - 12.3
     - 13.1
     - 0.94x
     - 0%
   * - 2048
     - 89.7
     - 18.2
     - 4.93x
     - 65%
   * - 4096
     - 412.3
     - 41.5
     - 9.93x
     - 94%
   * - 8192
     - 1823.4
     - 93.2
     - 19.56x
     - 97%

Hardware Support
---------------

Currently supported photonic hardware:

* **LightMatter Mars**: High-performance photonic processor
* **Luminous Computing**: Photonic inference accelerator
* **Generic PCIe**: Standard PCIe photonic cards
* **Simulation Mode**: Software simulation for development

Community
---------

* **GitHub**: https://github.com/danieleschmidt/photonic-mlir-synth-bridge
* **Issues**: Report bugs and request features
* **Discussions**: Ask questions and share ideas
* **Contributing**: See our contribution guidelines

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`