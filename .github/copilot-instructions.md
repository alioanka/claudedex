<!-- Copilot / AI agent instructions for immediate productivity in this repo -->
# ClaudeDex — AI Coding Agent Instructions

Purpose: give an AI coding agent the exact, actionable context needed to make safe, high-value changes quickly.

- Quick orientation:
  - Entrypoint: `main.py` (run with `python main.py --mode development`).
  - Key packages: see `requirements.txt` and `test-requirements.txt`.
  - Primary directories: `core/` (engine, event bus, decision-maker), `data/` (collectors/processors/storage), `analysis/` (token/market analysis), `ml/` (models), `trading/` (executors/strategies), `config/` (settings & validation), `scripts/` (ops helpers).

- Big-picture architecture (concise):
  - Data flows from `data/collectors/*` → `data/processors/*` → `analysis/*` and `ml/*` → `core/decision_maker.py` → `trading/*` executors. The `core/engine.py` orchestrates lifecycle (`start`/`stop`) and uses `core/event_bus.py` for internal events.
  - Persisted state: PostgreSQL/Timescale (models under `data/storage/models.py`) and Redis cache (`data/storage/cache.py`). DB bootstrapping lives in `scripts/setup_database.py`.

- Developer workflows / commands (Windows PowerShell):
  - Create venv & activate: `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
  - Install deps: `pip install -r requirements.txt`
  - Run app (dev): `python main.py --mode development`
  - Run tests: `pytest tests/` (or a subset `pytest tests/unit/`)
  - Coverage: `pytest --cov=. --cov-report=html`
  - Docker compose: `docker-compose up -d` (services described in `docker-compose.yml`)

- Project-specific conventions to follow (do not improvise):
  - Async-first: most long-running modules are `async def` (engine, collectors, analysis). Use `async/await` when integrating with those modules.
  - Type hints: prefer complete annotations for new public functions (this repo uses them broadly).
  - File placement: add new collectors under `data/collectors/`; processors under `data/processors/`; analysis logic under `analysis/`; ML models under `ml/models/` and reference them from `ml/` loader helpers.
  - Config surface: runtime flags are read by `config/settings.py` and JSON configs under `config/` (e.g., `config/trading.json`). Update both when adding features.
  - No secrets in repo: `.env.example` is the template. Never write actual keys into files.

- Integration points and external dependencies to be aware of:
  - Dex & chain data: DexScreener collectors (see `data/collectors/dexscreener.py`) and chain RPC usage (web3). Mock these in tests or use recorded fixtures under `tests/fixtures/`.
  - Execution & MEV: MEV protections and external executors exist in `trading/executors/` (e.g., `mev_protection.py`, `toxisol_api.py`) — avoid changing execution flow without running integration tests.
  - Storage: DB models in `data/storage/models.py` and connection logic in `data/storage/database.py` — migrations/scripts are used in `scripts/`.

- How to make a safe change (checklist):
  1. Run unit tests for affected modules: `pytest tests/unit/<module>`.
 2. Add/update fixtures under `tests/fixtures/` for collector or web3 responses.
 3. If adding dependencies, update `requirements.txt` and `setup.py`.
 4. Update documentation: short note in `docs/` and top-level `README.md` if behavior/commands change.
 5. For runtime changes, include a configuration key in `config/` and default in `config/settings.py`.

- Concrete examples to copy/paste when editing or testing:
  - Start engine locally (dev): `python main.py --mode development`
  - Run a single test file: `pytest tests/unit/test_engine.py -q`
  - Run python script helper: `python scripts/setup_database.py`

- CI and PR notes:
  - CI is configured via `.github/workflows/ci.yml`. Ensure linting/tests pass before proposing changes.
  - Keep changes narrowly scoped — many modules are interdependent (engine ↔ analysis ↔ trading).

- What the agent should not do automatically:
  - Do not add hard-coded API keys or secrets into files.
  - Do not change execution flows in `core/engine.py` or `trading/executors/*` without adding tests and a short design note in the PR.

- If anything here is unclear or you want more examples (e.g., how collectors are registered in the engine), request the specific area and the agent will expand with inline code examples taken from the repository.
