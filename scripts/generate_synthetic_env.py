"""CLI script to generate a synthetic digital payment environment for Finsheild.

Example usage:

    # CI scale (~5000 transactions, <30s)
    PYTHONPATH=src python scripts/generate_synthetic_env.py --scale ci --out data/synthetic_env/ci

    # Dev scale (~50k transactions)
    PYTHONPATH=src python scripts/generate_synthetic_env.py --scale dev --out data/synthetic_env/dev

    # Custom
    PYTHONPATH=src python scripts/generate_synthetic_env.py \
        --n-users 5000 --n-transactions 100000 --out data/synthetic_env/large --seed 1234

The output is a directory containing one ``.parquet`` file per table plus a
``metadata.json`` with the configuration and per-scenario breakdown.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src/ to sys.path so this script works without an editable install.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from finsheild.synthetic_env import SyntheticEnvConfig, generate_environment  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scale", choices=("ci", "dev"), default="dev",
                    help="Use pre-baked CI or dev scale defaults.")
    p.add_argument("--seed", type=int, default=None,
                    help="Override the seed (default: scale-specific).")
    p.add_argument("--n-users", type=int, default=None)
    p.add_argument("--n-accounts", type=int, default=None)
    p.add_argument("--n-devices", type=int, default=None)
    p.add_argument("--n-merchants", type=int, default=None)
    p.add_argument("--n-locations", type=int, default=None)
    p.add_argument("--n-transactions", type=int, default=None)
    p.add_argument("--time-span-days", type=int, default=None)
    p.add_argument("--fraud-rate", type=float, default=None)
    p.add_argument("--start-ts", type=str, default=None)
    p.add_argument("--out", type=Path, required=True,
                    help="Output directory (created if missing).")
    p.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if not args.quiet:
        logging.basicConfig(level=logging.INFO,
                             format="%(asctime)s %(levelname)s %(message)s")

    if args.scale == "ci":
        config = SyntheticEnvConfig.ci()
    else:
        config = SyntheticEnvConfig.dev()

    overrides = {
        "seed": args.seed,
        "n_users": args.n_users,
        "n_accounts": args.n_accounts,
        "n_devices": args.n_devices,
        "n_merchants": args.n_merchants,
        "n_locations": args.n_locations,
        "n_transactions": args.n_transactions,
        "time_span_days": args.time_span_days,
        "fraud_rate": args.fraud_rate,
        "start_ts": args.start_ts,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        config = SyntheticEnvConfig(**{**config.to_dict(), **overrides})

    if not args.quiet:
        logger.info("Generating environment with: %s", config.to_dict())
    env = generate_environment(config)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, df in env.tables.items():
        target = out_dir / f"{name}.{args.format}"
        if args.format == "parquet":
            df.to_parquet(target, index=False)
        else:
            df.to_csv(target, index=False)
        if not args.quiet:
            logger.info("wrote %s (%d rows)", target, len(df))

    meta = {
        "config": config.to_dict(),
        "tables": {n: {"rows": len(df)} for n, df in env.tables.items()},
        "scenario_breakdown": env.scenario_breakdown().to_dict(orient="records"),
        "fraud_rate": env.fraud_rate(),
        "format": args.format,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2,
                                                       default=str))
    if not args.quiet:
        logger.info("wrote %s", out_dir / "metadata.json")
        logger.info("fraud_rate=%.4f", env.fraud_rate())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())