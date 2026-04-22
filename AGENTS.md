# Repository Guidelines

## Project Structure & Module Organization
- `tradingagents/`: core package.
- `tradingagents/agents/`: role-specific agents (analysts, researchers, trader).
- `tradingagents/graph/`: orchestration and propagation flow.
- `tradingagents/dataflows/`: market/news/fundamental data adapters.
- `tradingagents/llm_clients/`: provider clients, model catalog, validators.
- `cli/`: Typer-based CLI entrypoint and UI helpers (`cli/main.py`).
- `tests/`: unit tests (`test_*.py`).
- `assets/`: diagrams and CLI screenshots for docs.
- Top-level examples: `main.py` and `test.py`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create/activate local env.
- `pip install .`: install package and CLI script (`tradingagents`).
- `python -m cli.main` or `tradingagents`: run interactive CLI.
- `python main.py`: run the sample programmatic workflow.
- `python -m unittest discover -s tests -p "test_*.py"`: run all tests.
- `docker compose run --rm tradingagents`: run in container.
- `docker compose --profile ollama run --rm tradingagents-ollama`: container run with Ollama profile.

## Coding Style & Naming Conventions
- Follow PEP 8 defaults: 4-space indentation, readable line lengths, and clear imports.
- Use `snake_case` for functions/variables/modules, `PascalCase` for classes, and `UPPER_CASE` for constants (for example `DEFAULT_CONFIG`).
- Keep provider/model validation logic centralized in `tradingagents/llm_clients/`.
- Add type hints when practical, especially for public helpers and client interfaces.

## Testing Guidelines
- Framework: built-in `unittest`.
- Naming: files `test_*.py`, classes `*Tests`/`Test*`, methods `test_*`.
- Add focused unit tests for new vendor/model behavior and config validation paths.
- Before opening a PR, run full discovery plus targeted tests for edited modules.

## Commit & Pull Request Guidelines
- Use Conventional Commit style seen in history: `feat: ...`, `fix: ...`, `refactor: ...`, `chore: ...`.
- Keep commits scoped to one logical change.
- PRs should include: concise summary, rationale, test evidence (command + result), and linked issue (if any).
- Include CLI screenshots only when changing visible terminal UX.

## Security & Configuration Tips
- Never commit secrets; use `.env`, `.env.enterprise`, and provided example files.
- Required keys vary by provider (for example `OPENAI_API_KEY`, `GOOGLE_API_KEY`).
- Prefer `TRADINGAGENTS_RESULTS_DIR` and `TRADINGAGENTS_CACHE_DIR` overrides for custom runtime paths.
