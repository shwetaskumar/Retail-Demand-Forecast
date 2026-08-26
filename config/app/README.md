# config/app — Databricks App (config CRUD front-end)

Placeholder. Per `docs/instructions.md` §20, config lives in a single EAV Delta table
(`control.forecast_config`), not YAML. This folder will hold the Databricks App
that provides a CRUD UI over that table so adding/editing a market is an
INSERT/UPDATE through the app, never a new file or a code change.

Not built yet. When built, it must:
- read/write the `control.forecast_config` table directly (same table `config/loader.py` reads);
- enforce the typed schema in `config/ddl/config_tables.sql`;
- require no deploy of pipeline code to onboard a market (docs/instructions.md §21 acceptance criterion 6).
