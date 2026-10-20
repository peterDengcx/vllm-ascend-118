# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Fast, single-pass GPQA Diamond evaluation against the local HY4 API.

Uses the public dataset referenced by openai/simple-evals/gpqa_eval.py.
This is a 0-shot direct-answer evaluation with reasoning_effort=no_think;
it must not be compared to published chain-of-thought or repeated-pass scores.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import regex as re
from validate import record

DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"


def evaluate(args):
    dataset_path = args.directory / "gpqa_diamond.csv"
    if not dataset_path.exists():
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
            dataset_path.write_bytes(response.read())
    data = dataset_path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    if args.limit:
        rows = rows[: args.limit]
    rng = random.Random(0)
    cases = []
    for index, row in enumerate(rows):
        choices = [row["Correct Answer"], *[row[f"Incorrect Answer {i}"] for i in range(1, 4)]]
        permutation = rng.sample(range(4), 4)
        prompt = (
            "Choose the correct answer to this multiple-choice question. "
            "Reply with exactly one letter: A, B, C, or D.\n\n"
            + row["Question"]
            + "\n\n"
            + "\n".join(f"{letter}. {choices[choice]}" for letter, choice in zip("ABCD", permutation, strict=True))
        )
        cases.append((index, prompt, "ABCD"[permutation.index(0)]))
    run_id = uuid.uuid4().hex
    settings = {
        "run_id": run_id,
        "dataset_url": DATASET_URL,
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "samples": len(cases),
        "repeats": 1,
        "seed": 0,
        "concurrency": args.concurrency,
        "max_tokens": 1,
        "temperature": 0,
        "reasoning_effort": "no_think",
        "method": "0-shot direct answer, constrained choice",
        "output_choices": list("ABCD"),
    }
    state = args.directory / "server.json"
    if state.exists():
        server = json.loads(state.read_text())
        settings["server_stage"] = server["stage"]
        settings["server_command"] = server["command"]
    record(args.directory, "gpqa_start", settings, "Started", "started")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def sample(case):
        index, prompt, correct = case
        body = {
            "model": "hy4",
            "messages": [
                {
                    "role": "system",
                    "content": "Choose the correct option. Reply with exactly one uppercase letter: A, B, C, or D.",
                },
                {"role": "user", "content": prompt},
            ],
            "structured_outputs": {"choice": list("ABCD")},
            "max_tokens": 1,
            "temperature": 0,
            "chat_template_kwargs": {"reasoning_effort": "no_think"},
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        response = None
        error = None
        content = ""
        try:
            with opener.open(request, timeout=600) as stream:
                response = json.load(stream)
            content = response["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as exc:
            response = {"http_error_body": exc.read().decode("utf-8", errors="replace")}
            error = repr(exc)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            error = repr(exc)
        match = re.fullmatch(r"\s*(?:Answer\s*:\s*)?\(?([A-D])\)?[.。]?\s*", content, re.IGNORECASE)
        predicted = match.group(1).upper() if match else None
        result = {
            "run_id": run_id,
            "index": index,
            "correct_answer": correct,
            "predicted_answer": predicted,
            "correct": predicted == correct,
            "response": response,
            "error": error,
            "seconds": time.monotonic() - started,
        }
        return body, result

    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(sample, case) for case in cases]
        for future in as_completed(futures):
            body, result = future.result()
            results.append(result)
            record(args.directory, "gpqa_sample", body, result, "completed" if not result["error"] else "failed")
    summary = {
        **settings,
        "correct": sum(result["correct"] for result in results),
        "unparsed": sum(result["predicted_answer"] is None for result in results),
        "request_errors": sum(result["error"] is not None for result in results),
        "prompt_tokens": sum((result["response"] or {}).get("usage", {}).get("prompt_tokens", 0) for result in results),
        "completion_tokens": sum(
            (result["response"] or {}).get("usage", {}).get("completion_tokens", 0) for result in results
        ),
        "seconds": time.monotonic() - started,
    }
    summary["accuracy"] = summary["correct"] / len(cases)
    record(
        args.directory, "gpqa_complete", settings, summary, "completed" if not summary["request_errors"] else "failed"
    )
    (args.directory / "gpqa-summary.json").write_text(json.dumps(summary, indent=2))
    if summary["request_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(".vllm_ascend/hy4_adaptation"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int)
    evaluate(parser.parse_args())
