from dataclasses import dataclass

@dataclass
class TransformationConfig:
    """Configuration for the transformation stage."""
    test_size: float = 0.2
    val_size: float = 0.5
    random_state: int = 42
    target_column: str = "Class"