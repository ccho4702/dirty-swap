"""Resumable GPU gate for one common QA criterion across question types."""

import argparse
import json
import time
import uuid
from pathlib import Path

from dirty_swapping.backend import TransformersBackend
from dirty_swapping.config import load_spec, runtime_spec
from dirty_swapping.core import atomic_json, digest, file_digest, now, source_fingerprint
from dirty_swapping.preference import ProbePreferenceJudge
from dirty_swapping.runner import _heartbeat, runtime_identity


def cases():
    problems = [
        (
            "The supplied passage says Project Sable launched in 2018. What year did it launch?",
            "I must use the supplied passage.",
            "Project Sable launched in 2020.",
            "The passage states 2018, so the answer is 2018.",
        ),
        (
            "Context: mitochondrial ATP synthase uses a proton gradient to synthesize ATP. What drives ATP synthesis in this description?",
            "I must identify the stated cause.",
            "Light directly creates glucose, which is the stated driver.",
            "The supplied proton gradient drives ATP synthase to produce ATP.",
        ),
        (
            "Context: Nora was born in Lisbon. Lisbon is in Portugal. What country was Nora born in?",
            "I must connect the two supplied facts.",
            "Nora was born in Spain.",
            "Nora was born in Lisbon, which is in Portugal; therefore Portugal.",
        ),
        (
            "23 apples are present and 7 are eaten. How many remain?",
            "I subtract the eaten apples.",
            "23 minus 7 equals 17.",
            "23 minus 7 equals 16, so 16 remain.",
        ),
    ]
    result = []
    for index, (question, history, wrong, correct) in enumerate(problems):
        for group, original, alternative, expected in (
            ("correct", wrong, correct, True),
            ("harm", correct, wrong, False),
        ):
            result.append(
                {
                    "id": f"{group}-{index}",
                    "question": question,
                    "history": history,
                    "original": original,
                    "alternative": alternative,
                    "expected": expected,
                }
            )
        if index < 4:
            result.append(
                {
                    "id": f"tie-{index}",
                    "question": question,
                    "history": history,
                    "original": correct,
                    "alternative": correct,
                    "expected": False,
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/general-probe.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("."))
    parser.add_argument("--run-name")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    spec = runtime_spec(load_spec(args.config), work)
    items = cases()
    identity = {
        "kind": "preference-judge-validation",
        "config": spec,
        "source": source_fingerprint(),
        "script": file_digest(Path(__file__)),
        "runtime": runtime_identity(),
        "cases_sha256": digest(items),
    }
    if args.resume:
        root = args.resume.resolve()
        if json.loads((root / "identity.json").read_text()) != identity:
            raise ValueError("judge gate identity changed; resume rejected")
    else:
        name = (
            args.run_name
            or time.strftime("judge-%Y%m%dT%H%M%S-", time.gmtime()) + uuid.uuid4().hex[:6]
        )
        if not name or Path(name).name != name:
            raise ValueError("run-name must be a directory name")
        root = work / "outputs" / "judge-checks" / name
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / "identity.json", identity)
    logs = work / "logs" / "judge-checks" / root.name
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "run.log").open("a", buffering=1) as stream:

        def log(message):
            line = f"{now()} {message}"
            print(line, flush=True)
            stream.write(line + "\n")

        results = []
        atomic_json(root / "status.json", {"state": "running", "at": now()})
        try:
            with _heartbeat("loading judge model", log, 15):
                backend = TransformersBackend(spec["model"])
            judge = ProbePreferenceJudge(backend, spec["generation"]["probe"])
            started = time.monotonic()
            for index, item in enumerate(items, 1):
                path = root / "cases" / (item["id"] + ".json")
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved["sha256"] != digest(saved["payload"]):
                        raise ValueError("corrupt judge checkpoint")
                    result = saved["payload"]
                else:
                    # Expected labels are used only after the judge returns.
                    decision = judge.compare(
                        item["question"], item["history"], item["original"], item["alternative"]
                    )
                    result = {
                        "case": item,
                        "decision": decision,
                        "passed": decision["accepted"] == item["expected"],
                    }
                    atomic_json(path, {"payload": result, "sha256": digest(result)})
                results.append(result)
                elapsed = time.monotonic() - started
                log(
                    f"{index}/{len(items)} {item['id']} passed={result['passed']}; elapsed={elapsed:.0f}s ETA={elapsed / index * (len(items) - index):.0f}s"
                )
        except BaseException:
            atomic_json(root / "status.json", {"state": "interrupted", "at": now()})
            raise
        groups = {}
        for group in ("correct", "harm", "tie"):
            selected = [r for r in results if r["case"]["id"].startswith(group + "-")]
            groups[group] = {"n": len(selected), "passed": sum(r["passed"] for r in selected)}
        summary = {
            "at": now(),
            "groups": groups,
            "passed": all(result["passed"] for result in results),
            "judge_seconds": sum(r["decision"]["seconds"] for r in results),
        }
        atomic_json(root / "summary.json", summary)
        atomic_json(root / "status.json", {"state": "complete", "at": now()})
        log(json.dumps(summary))


if __name__ == "__main__":
    main()
