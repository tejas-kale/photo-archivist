from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SourceMedia:
    source: str
    source_id: str
    path: Path
    metadata: dict = field(default_factory=dict)
