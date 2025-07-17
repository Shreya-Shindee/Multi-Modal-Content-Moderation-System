#!/usr/bin/env python3
"""
Automatic code formatting script to fix PEP 8 issues.
This script fixes common formatting issues like trailing whitespace,
line length, etc.
"""

import sys
from pathlib import Path


def fix_trailing_whitespace(content: str) -> str:
    """Remove trailing whitespace from all lines."""
    lines = content.split("\n")
    fixed_lines = [line.rstrip() for line in lines]
    return "\n".join(fixed_lines)


def fix_line_length(content: str, max_length: int = 79) -> str:
    """Fix lines that are too long by breaking them appropriately."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        if len(line) <= max_length:
            fixed_lines.append(line)
            continue

        # Handle f-strings and long print statements
        if 'print(f"' in line and len(line) > max_length:
            # Break long f-string prints
            indent = len(line) - len(line.lstrip())
            if "Processing time:" in line:
                fixed_lines.append(
                    " " * indent + 'print(f"   Processing time: "'
                )
                fixed_lines.append(
                    " " * indent
                    + "f\"{test_result.get('processing_time', 0):.3f}s\")"
                )
            else:
                fixed_lines.append(line)
        elif "curl -X POST" in line:
            # Don't break curl commands in comments/strings
            fixed_lines.append(line)
        else:
            # For other long lines, try to break at logical points
            if "," in line and not line.strip().startswith("#"):
                # Break at commas
                parts = line.split(",")
                if len(parts) > 1:
                    indent = len(line) - len(line.lstrip())
                    first_part = parts[0] + ","
                    fixed_lines.append(first_part)
                    for part in parts[1:-1]:
                        fixed_lines.append(
                            " " * (indent + 4) + part.strip() + ","
                        )
                    if parts[-1].strip():
                        fixed_lines.append(
                            " " * (indent + 4) + parts[-1].strip()
                        )
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_blank_lines(content: str) -> str:
    """Fix blank line issues around function/class definitions."""
    lines = content.split("\n")
    fixed_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for function/class definitions that need 2 blank lines before
        if (
            line.strip().startswith("def ")
            or line.strip().startswith("class ")
        ) and i > 0:
            # Don't add blank lines if we're inside a class (indented method)
            if not line.startswith("    ") or line.strip().startswith(
                "class "
            ):
                # Check how many blank lines we have before
                blank_count = 0
                j = i - 1
                while j >= 0 and not lines[j].strip():
                    blank_count += 1
                    j -= 1

                # Remove existing blank lines and add exactly 2
                while fixed_lines and not fixed_lines[-1].strip():
                    fixed_lines.pop()

                # Add 2 blank lines before top-level functions/classes
                if j >= 0 and not lines[j].startswith(
                    "    "
                ):  # Previous line wasn't indented
                    fixed_lines.extend(["", ""])

        fixed_lines.append(line)
        i += 1

    return "\n".join(fixed_lines)


def fix_imports(content: str) -> str:
    """Remove unused imports."""
    lines = content.split("\n")

    # Track which imports are actually used
    import_lines = []
    code_lines = []

    in_imports = True
    for line in lines:
        if line.strip().startswith(("import ", "from ")) and in_imports:
            import_lines.append(line)
        else:
            if line.strip() and not line.strip().startswith("#"):
                in_imports = False
            code_lines.append(line)

    # Check which imports are used
    all_code = "\n".join(code_lines)
    used_imports = []

    for import_line in import_lines:
        if "import " in import_line:
            # Extract module names
            if import_line.strip().startswith("from "):
                # from module import name
                parts = import_line.strip().split()
                if len(parts) >= 4:
                    module = parts[1]
                    imports = " ".join(parts[3:])
                    # Check if any imported names are used
                    for imp in imports.split(","):
                        imp = imp.strip()
                        if imp in all_code:
                            used_imports.append(import_line)
                            break
            else:
                # import module
                module = import_line.strip().replace("import ", "").split()[0]
                if module in all_code:
                    used_imports.append(import_line)

    # Rebuild content with used imports
    result_lines = (
        used_imports + [""] + code_lines if used_imports else code_lines
    )
    return "\n".join(result_lines)


def fix_indentation(content: str) -> str:
    """Fix indentation issues."""
    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        if line.strip():
            # Fix continuation line indentation
            if i > 0 and lines[i - 1].rstrip().endswith(("(", ",", "\\")):
                # This is a continuation line
                prev_line = lines[i - 1]
                prev_indent = len(prev_line) - len(prev_line.lstrip())

                # Ensure proper indentation for continuation
                if not line.startswith(" " * (prev_indent + 4)):
                    stripped = line.lstrip()
                    line = " " * (prev_indent + 4) + stripped

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_file(file_path: Path) -> bool:
    """Fix formatting issues in a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_trailing_whitespace(content)
        content = fix_line_length(content)
        content = fix_blank_lines(content)
        content = fix_indentation(content)

        # Only fix imports for non-demo files to avoid breaking functionality
        if "demo.py" not in str(file_path):
            content = fix_imports(content)

        # Write back only if changed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            return True
        else:
            print(f"⏭️ No changes needed: {file_path}")
            return False

    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False


def main():
    """Main function to fix all Python files in the project."""
    project_root = Path(__file__).parent

    # Files to fix
    python_files = [
        "preprocessing/text_processor.py",
        "preprocessing/image_processor.py",
        "models/text_model.py",
        "models/image_model.py",
        "models/multimodal_model.py",
        "demo.py",
        "simple_demo.py",
        "system_status.py",
    ]

    print("🔧 Starting automatic code formatting...")
    print("=" * 50)

    fixed_count = 0
    for file_path in python_files:
        full_path = project_root / file_path
        if full_path.exists():
            if fix_file(full_path):
                fixed_count += 1
        else:
            print(f"⚠️ File not found: {full_path}")

    print("=" * 50)
    print(f"🎉 Formatting complete! Fixed {fixed_count} files.")

    # Run a quick validation
    print("\n🧪 Running quick validation...")
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile"]
            + [
                str(project_root / f)
                for f in python_files
                if (project_root / f).exists()
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            print("✅ All files compile successfully!")
        else:
            print(f"⚠️ Compilation warnings/errors: {result.stderr}")
    except Exception as e:
        print(f"⚠️ Could not run validation: {e}")


if __name__ == "__main__":
    main()
