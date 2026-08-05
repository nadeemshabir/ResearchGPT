"""ResearchGPT: retrieval-augmented question answering over research papers.

Submodules are not imported here on purpose. ``src.generation`` pulls in torch
and sentence-transformers, which costs several seconds; importing
``src.config`` or ``src.exceptions`` should stay instant.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
