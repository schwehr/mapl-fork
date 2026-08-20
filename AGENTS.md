# Agent Guidelines for MAPL Repository

## Code & Docstring Style

- **Docstrings**:
  - **CRITICAL RULE**: All module, class, method, and function docstrings must
    strictly follow **Standard Google Python Docstring Style**.
  - Include clearly formatted `Args:`, `Returns:`, `Raises:`, `Yields:`, and
    `Attributes:` sections as applicable.
  - Avoid unstructured, verbose, or legacy docstring formatting.
- **String Formatting**:
  - Always use modern Python **f-strings** (`f"Value: {val}"`) for string
    concatenation and formatting. Never use legacy `%` formatting or
    `.format()`.
- **Type Annotations**:
  - Provide precise, tight type annotations for all function signatures and
    return types.
  - Avoid generic `Any` types; prefer specific types such as `Sequence[int]`,
    `Buffer`, `Self`, or `Literal`.
  - Avoid explicit `Union`/`Optional` types. Use '|'.

## 7. Version Control & Commit Messages

- **Feature Branches**:
  - **CRITICAL RULE**: All code changes and refactoring work MUST be performed
    on dedicated git feature branches (e.g., `git checkout -b <branch-name>`).
  - Never make direct commits on the `main` branch.
- **Code Review**:
  - Always do a code review before committing.
  - Use a different LLM model for the subagent doing the review.
  - Create 1-3 suggestions for improvement to the code based on the current changes.
  - See if there needs to be any changes to `AGENTS.md` based on the current
    changes and propose improvements.
- **Conventional Commits**:
  - All git commit messages MUST adhere to the **Conventional Commits**
    specification (`<type>(<optional scope>): <subject>`).
  - Examples:
    - `feat(dunder): enable the __foo__ feature`
    - `refactor(tests): switch test_init.py from unittest to pytest`
    - `chore(license): Add the SPDX header`
    - `docs: import legacy manuals into docs/ directory`
- **NO Tag or Conversation ID Entries**:
  - **CRITICAL RULE**: Commit messages must **NEVER** contain `TAG=` or `CONV=`
    lines or entries. These are reserved for internal Piper/CL tools and must be
    omitted from all git commits in this repository.

## Package Management and Lockfile Updates

When resolving dependencies or updating `uv.lock`, always use the public PyPI index (`https://pypi.org/simple`).

- Always supply `--default-index https://pypi.org/simple` when running
  `uv lock`, `uv add`, or related commands that touch `uv.lock`.
- Note that `pyproject.toml` is configured with:
  ```toml
  [[tool.uv.index]]
  name = "pypi"
  url = "https://pypi.org/simple"
  default = true
  ```
- If running in an internal development environment where pip/system config
  specifies an internal proxy mirror (such as `airlock-proxy`), prepend
  `PIP_CONFIG_FILE=/dev/null` or pass `--default-index https://pypi.org/simple`
  so internal mirror URLs are never written into `uv.lock`.
