"""Unit conversion helpers."""

BYTES_PER_KIB = 1024
BYTES_PER_MIB = 1024 * BYTES_PER_KIB
BYTES_PER_GIB = 1024 * BYTES_PER_MIB
BYTES_PER_TIB = 1024 * BYTES_PER_GIB


def gbps_to_bytes_per_us(gbps: float) -> float:
    """Convert decimal GB/s to bytes/us."""
    return gbps * 1_000_000_000 / 1_000_000


def bytes_to_gib(value: float) -> float:
    return value / BYTES_PER_GIB


def bytes_to_gb(value: float) -> float:
    return value / 1_000_000_000


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return (value + divisor - 1) // divisor

