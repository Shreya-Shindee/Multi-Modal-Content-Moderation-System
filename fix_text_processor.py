#!/usr/bin/env python3
"""
Fix specific formatting issues in text_processor.py
"""


def fix_text_processor():
    """Fix formatting issues in text_processor.py"""
    file_path = "preprocessing/text_processor.py"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix trailing whitespace and blank lines with whitespace
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()
        fixed_lines.append(line)

    # Join back and fix specific line length issues
    content = "\n".join(fixed_lines)

    # Fix the specific long line
    content = content.replace(
        '            df["text_length"] = df["cleaned_text"].str.len()',
        '            df["text_length"] = df["cleaned_text"].str.len()',
    )

    # Write back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("Fixed formatting issues in text_processor.py")


if __name__ == "__main__":
    fix_text_processor()
