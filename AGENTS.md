# AGENTS.md

## Dependency policy

Do not install any library, package, dependency, or tool in this project without the user's explicit permission.

This includes:
- `pip install`
- `uv add`
- `uv pip install`
- `poetry add`
- `poetry install`
- `conda install`
- `npm install`
- `pnpm install`
- `yarn add`
- `brew install`

If a task appears to require a new dependency, ask the user for permission before installing anything.

## Python execution

Use `uv` to run Python scripts and Python modules in this project.
