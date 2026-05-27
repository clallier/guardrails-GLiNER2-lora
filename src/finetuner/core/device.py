"""Module for resolving PyTorch hardware acceleration devices dynamically."""

import logging

import torch

logger = logging.getLogger(__name__)


class DeviceResolver:
    """Resolves and configures the optimal execution device for PyTorch operations.

    High level role: Hardware target abstraction layer.
    Description: Provides static utility interfaces to determine the optimal active device
    (CUDA, Apple Silicon MPS, or CPU).
    """

    @staticmethod
    def get_optimal_device() -> str:
        """Resolves the most suitable hardware device available for PyTorch.

        High level role: Hardware device resolver.
        Description: Inspects systemic runtime capability for CUDA or Apple Silicon MPS support
        and falls back to CPU if no hardware accelerators are present.

        Args:
            None.

        Returns:
            str: Resolved PyTorch device string ("cuda", "mps", or "cpu").

        Raises:
            None.

        Examples:
            >>> device_name = DeviceResolver.get_optimal_device()
            >>> print(device_name)
            'mps'
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
