import torch
from pathlib import Path

try:
    from . import _C, ops
except (ImportError, SystemError):
    # Absolute import fallback (not ideal, but helps for direct script execution)
    import extension_cpp._C as _C
    import extension_cpp.ops as ops