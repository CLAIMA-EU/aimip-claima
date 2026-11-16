"""Custom exception hierarchy for the AIMIP Playground CLI."""


class AimipError(Exception):
    """Base exception for all expected user-facing AIMIP CLI errors."""


class ValidationError(AimipError):
    """Raised when CLI input, environment values, or file contents fail validation."""


class RegistryError(AimipError):
    """Raised when the model registry cannot be loaded or is malformed."""


class ConfigError(AimipError):
    """Raised when run/spin configuration constraints are violated."""
