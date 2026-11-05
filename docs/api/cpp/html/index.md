# C++ API Reference

The C++ API is documented via **Doxygen**.

## Generate

```bash
cd tile_compile_cpp
doxygen Doxyfile
```

Requires `doxygen` and `graphviz` (for diagrams):

```bash
# Ubuntu/Debian
sudo apt install doxygen graphviz

# macOS
brew install doxygen graphviz

# Windows
choco install doxygen.portable graphviz
```

## Output

HTML documentation is generated to `docs/api/cpp/html/`.

Open `docs/api/cpp/html/index.html` in a browser.

## Coverage

- Core pipeline (`src/`)
- Runner applications (`apps/`)
- Public headers (`include/`)

Key namespaces:
- `tile_compile::core` — pipeline phases
- `tile_compile::registration` — alignment
- `tile_compile::reconstruction` — CFA forward drizzle, quality maps, and multiband fusion
