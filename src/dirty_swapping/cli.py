"""Setup, paired evaluation, recovery, reporting, and local submission checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_spec, runtime_spec
from .data import download_datasets, download_models, prepare, sources
from .runner import create_run, execute, report
from .scoring import load_jsonl, score


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dirty-swapping")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("config", help="print the packaged default configuration")
    for name in ("download", "prepare", "setup", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--work-dir", type=Path, default=Path("."))
        command.add_argument("--model-cache", type=Path)
        command.add_argument("--device", help="e.g. cuda:0 or cpu")
        command.add_argument("--datasets", nargs="+", choices=list(sources()))
        if name == "download":
            command.add_argument("--data-only", action="store_true")
        if name == "run":
            command.add_argument("--limit", type=int)
            command.add_argument("--run-name")
            command.add_argument(
                "--split", choices=("development", "evaluation"), default="evaluation"
            )
    for name in ("resume", "report", "status"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
    local_score = commands.add_parser(
        "score", help="check prediction JSONL against labeled development data"
    )
    local_score.add_argument("--gold", type=Path, required=True)
    local_score.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "config":
        print(json.dumps(load_spec(), indent=2))
        return
    if args.command == "score":
        print(json.dumps(score(load_jsonl(args.gold), load_jsonl(args.predictions)), indent=2))
        return
    if args.command in ("resume", "report", "status"):
        run = args.run.expanduser().resolve()
        if args.command == "resume":
            print(json.dumps(execute(run, run.parent.parent), indent=2))
        elif args.command == "report":
            print(json.dumps(report(run), indent=2))
        else:
            print((run / "status.json").read_text())
        return

    work_dir = args.work_dir.expanduser().resolve()
    spec = runtime_spec(
        load_spec(args.config), work_dir, device=args.device, model_cache=args.model_cache
    )
    datasets = args.datasets or spec["data"]["default_datasets"]
    if args.command in ("download", "setup"):
        download_datasets(work_dir, datasets)
        if not getattr(args, "data_only", False):
            download_models(spec)
    if args.command in ("prepare", "setup"):
        prepare(work_dir, spec, datasets)
    if args.command == "run":
        run = create_run(
            work_dir, spec, datasets, limit=args.limit, run_name=args.run_name, split=args.split
        )
        print(f"Run: {run}", flush=True)
        print(json.dumps(execute(run, work_dir), indent=2))


if __name__ == "__main__":
    main()
