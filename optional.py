import nbformat

notebook_path = "notebooks/03_rag_librarian.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Remove broken widget metadata
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]

with open(notebook_path , "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Fixed notebook saved.")
