from pathlib import Path

import yaml


def write(media, description):
    path = Path(f"{media.path}.description.md")
    frontmatter = {
        "source": media.source,
        "source_id": media.source_id,
        "original_path": str(media.path),
        **media.metadata,
    }
    text = f"---\n{yaml.safe_dump(frontmatter, sort_keys=True)}---\n\n{description}\n"
    path.write_text(text)
    return path
