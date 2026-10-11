# Project Rules & Guidelines

## File Naming Conventions
- **No Numbers in Filenames**: Never include digits or numeric prefixes/suffixes in file names (e.g., use `principal_component_analysis.py` instead of `23_principal_component_analysis.py`).
- **Text-Only Names**: Use only meaningful, descriptive words and standard lowercase `snake_case` for Python source files, modules, and scripts.
- **Single Source of Truth**:
  - Do not create duplicate files, alias wrappers, or stub files (e.g., do not create duplicate wrappers for typos or session numbering).
  - When modifying or refactoring modules, update the existing file in place rather than creating duplicate copies.
  - Store outputs uniformly in `Output/plots/` and `Output/Report/` without generating redundant session-specific duplicate directories.
