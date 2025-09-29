import os

# Folders to skip
EXCLUDED_DIRS = {'.git', '.hg', '.svn', '.venv', 'venv', 'env', '__pycache__', 'node_modules', 'dist', 'build', 'logs'}

# File extensions to include (add more if needed)
INCLUDED_EXTS = {
    '.py', '.js', '.ts', '.html', '.css', '.scss', '.md', '.json',
    '.yaml', '.yml', '.toml', '.ini', '.txt', '.java', '.cpp', '.c', '.h'
}

def count_loc_in_file(file_path):
    """Count non-empty lines in a single file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"⚠️ Skipping {file_path}: {e}")
        return 0

def count_project_loc(root_dir="."):
    total_lines = 0
    file_count = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude unwanted directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in INCLUDED_EXTS:
                file_path = os.path.join(dirpath, filename)
                lines = count_loc_in_file(file_path)
                total_lines += lines
                file_count += 1

    return file_count, total_lines

if __name__ == "__main__":
    root = "."  # Change if you want another folder
    files, lines = count_project_loc(root)
    print(f"✅ Counted {lines:,} lines of code across {files} files (excluding {', '.join(EXCLUDED_DIRS)})")
