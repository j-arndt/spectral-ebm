# Contributing

Contributions should preserve the correctness-first scope of the project.

Before opening a pull request:

1. Run `python -m pytest` with third-party pytest plugin autoload disabled if your environment has unrelated plugins.
2. Run `ruff check .`.
3. Add or update tests for mathematical behavior, not only code paths.
4. Include benchmark configuration and raw output for performance claims.
5. Add citations and license notices for external algorithms, data, or code.

Do not submit claims of novelty, formal-proof correctness, or universal speedup without a documented comparison and evidence.
