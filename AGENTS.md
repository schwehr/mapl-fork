# Agent Guidelines for MAPL Repository

## Package Management and Lockfile Updates

When resolving dependencies or updating `uv.lock`, always use the public PyPI index (`https://pypi.org/simple`).

- Always supply `--default-index https://pypi.org/simple` when running `uv lock`, `uv add`, or related commands that touch `uv.lock`.
- Note that `pyproject.toml` is configured with:
  ```toml
  [[tool.uv.index]]
  name = "pypi"
  url = "https://pypi.org/simple"
  default = true
  ```
- If running in an internal development environment where pip/system config specifies an internal proxy mirror (such as `airlock-proxy`), prepend `PIP_CONFIG_FILE=/dev/null` or pass `--default-index https://pypi.org/simple` so internal mirror URLs are never written into `uv.lock`.
