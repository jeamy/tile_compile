#!/usr/bin/env python3
"""Remove all ![](img/...) image links from markdown files."""

import re
from pathlib import Path


def remove_image_links(md_path: Path) -> bool:
    """Remove all image links from a markdown file. Returns True if changes were made."""
    content = md_path.read_text(encoding="utf-8")
    original = content
    
    # Remove lines that contain only image links like ![](img/...)
    content = re.sub(r'^!\[\]\(img/[^)]+\)\s*$', '', content, flags=re.MULTILINE)
    
    # Remove inline image links
    content = re.sub(r'!\[\]\(img/[^)]+\)', '', content)
    
    # Clean up multiple consecutive blank lines (more than 2)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    if content != original:
        md_path.write_text(content, encoding="utf-8")
        return True
    return False


def main():
    base = Path(__file__).resolve().parent
    md_files = sorted([p for p in base.glob("*.md") if p.is_file()])
    
    changed = 0
    for md in md_files:
        if remove_image_links(md):
            print(f"Cleaned: {md.name}")
            changed += 1
    
    print(f"\nRemoved image links from {changed} file(s)")


if __name__ == "__main__":
    main()
