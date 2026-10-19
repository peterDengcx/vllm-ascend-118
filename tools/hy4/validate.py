# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Record HY4 deployment stages and actual curl requests in one JSONL log."""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def record(directory, stage, inputs, outputs, status):
    entry = {
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "stage": stage,
        "status": status,
        "input": inputs,
        "output": outputs,
    }
    with (directory / "validation.log").open("a") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def state_path(args):
    name = args.stage if args.stage.startswith("pd") else "server"
    return args.directory / f"{name}.json"


def launch(args):
    is_pd = args.stage.startswith("pd_")
    command = [
        "vllm",
        "serve",
        args.model,
        "--served-model-name",
        "hy4",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "4" if is_pd else "8",
        "--enable-expert-parallel",
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "512",
        "--kv-cache-memory-bytes",
        "2147483648",
        "--gpu-memory-utilization",
        "0.95",
        "--offload-backend",
        "prefetch",
        "--offload-group-size",
        "3" if is_pd else "5",
        "--offload-num-in-group",
        "2" if is_pd else "1",
        "--offload-prefetch-step",
        "1",
        "--offload-params",
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "--additional-config",
        '{"weight_nz_mode":0,"enable_cpu_binding":false}',
        "--async-scheduling" if args.stage != "eager" else "--no-async-scheduling",
    ]
    if args.stage in ("eager", "async"):
        command.append("--enforce-eager")
    else:
        command += [
            "--cudagraph-metrics",
            "--compilation-config",
            '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}',
        ]
    if args.stage == "mtp" or is_pd:
        command += ["--speculative-config", '{"method":"mtp","num_speculative_tokens":1}']
    if is_pd:
        producer = args.stage == "pd_prefill"
        transfer = {
            "kv_connector": "MooncakeConnectorV1",
            "kv_buffer_device": "npu",
            "kv_role": "kv_producer" if producer else "kv_consumer",
            "kv_parallel_size": 1,
            "kv_port": 20001 if producer else 20002,
            "kv_rank": 0 if producer else 1,
            "kv_connector_extra_config": {
                "prefill": {"tp_size": 4, "dp_size": 1},
                "decode": {"tp_size": 4, "dp_size": 1},
            },
        }
        command += ["--kv-transfer-config", json.dumps(transfer)]
    environment = dict(os.environ)
    environment.update(
        ASCEND_RT_VISIBLE_DEVICES=("0,1,2,3" if args.stage == "pd_prefill" else "4,5,6,7")
        if is_pd
        else "0,1,2,3,4,5,6,7",
        HCCL_OP_EXPANSION_MODE="AIV",
        HCCL_NPU_SOCKET_PORT_RANGE="auto",
        HCCL_HOST_SOCKET_PORT_RANGE="auto",
    )
    logfile = args.directory / f"{args.stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-server.log"
    with logfile.open("w") as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
    state = {"pid": process.pid, "stage": args.stage, "command": command, "log": str(logfile), "port": args.port}
    state_path(args).write_text(json.dumps(state, indent=2))
    record(args.directory, args.stage + "_launch", command, state, "started")


def readiness(args):
    state = json.loads(state_path(args).read_text())
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "curl",
                "--noproxy",
                "*",
                "-fsS",
                "--max-time",
                "3",
                f"http://127.0.0.1:{state['port']}" + ("/healthcheck" if state["stage"] == "pd" else "/v1/models"),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            record(args.directory, state["stage"] + "_ready", state["command"], result.stdout, "passed")
            return
        try:
            os.kill(state["pid"], 0)
            command = Path(f"/proc/{state['pid']}/cmdline").read_bytes()
            if not command:
                raise ProcessLookupError
        except (ProcessLookupError, FileNotFoundError):
            output = Path(state["log"]).read_text(errors="replace")
            record(args.directory, state["stage"] + "_ready", state["command"], output, "failed")
            raise SystemExit(1)
        time.sleep(3)
    record(
        args.directory,
        state["stage"] + "_ready",
        state["command"],
        Path(state["log"]).read_text(errors="replace"),
        "timeout",
    )
    raise SystemExit(1)


def smoke(args):
    state = json.loads(state_path(args).read_text())
    tests = [
        ("1+1等于几？只回答数字。", "2"),
        ("中国的首都是哪里？只回答城市名称。", "北京"),
        ("What is the capital of France? Answer only the city name.", "Paris"),
        ("请用中文写一句简短的自我介绍。", None),
        ("这是用于检查长上下文的填充文字。" * 300 + "中国的首都是哪里？只回答城市名称。", "北京"),
    ]

    def request(test):
        prompt, expected = test
        body = {
            "model": "hy4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 96,
            "chat_template_kwargs": {"reasoning_effort": "no_think"},
        }
        command = [
            "curl",
            "--noproxy",
            "*",
            "-sS",
            "--max-time",
            "300",
            "-w",
            "\n%{http_code}",
            f"http://127.0.0.1:{state['port']}/v1/chat/completions",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            json.dumps(body, ensure_ascii=False),
        ]
        started = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True)
        response, _, code = result.stdout.rpartition("\n")
        passed = False
        try:
            decoded = json.loads(response)
            content = decoded["choices"][0]["message"]["content"] or ""
            passed = (
                code == "200"
                and bool(content.strip())
                and "\ufffd" not in content
                and (expected is None or expected.lower() in content.lower())
            )
        except (ValueError, KeyError, IndexError):
            pass
        return (
            body,
            {
                "response": response,
                "http_code": code,
                "stderr": result.stderr,
                "seconds": time.monotonic() - started,
            },
            passed,
        )

    failures = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for body, output, passed in executor.map(request, tests):
            record(args.directory, state["stage"] + "_curl", body, output, "passed" if passed else "failed")
            failures += not passed
    if failures:
        raise SystemExit(1)


def stop(args):
    state = json.loads(state_path(args).read_text())
    with contextlib.suppress(ProcessLookupError):
        os.killpg(state["pid"], signal.SIGTERM)
    record(args.directory, state["stage"] + "_stop", {"process_group": state["pid"]}, "SIGTERM sent", "completed")


def run_command(args):
    started = time.monotonic()
    result = subprocess.run(args.command, capture_output=True, text=True)
    record(
        args.directory,
        args.label,
        args.command,
        {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "seconds": time.monotonic() - started,
            "exit_code": result.returncode,
        },
        "passed" if result.returncode == 0 else "failed",
    )
    raise SystemExit(result.returncode)


def metrics(args):
    state = json.loads(state_path(args).read_text())
    port = 8200 if args.stage == "pd" else state["port"]
    command = ["curl", "--noproxy", "*", "-fsS", "--max-time", "10", f"http://127.0.0.1:{port}/metrics"]
    result = subprocess.run(command, capture_output=True, text=True)
    selected = [
        line
        for line in result.stdout.splitlines()
        if not line.startswith("#") and any(key in line for key in ("spec_decode", "cudagraph", "kv_transfer"))
    ]
    (args.directory / f"{args.stage}-metrics.txt").write_text(result.stdout)
    record(
        args.directory,
        args.stage + "_metrics",
        command,
        {"metrics": selected, "stderr": result.stderr},
        "passed" if result.returncode == 0 else "failed",
    )
    raise SystemExit(result.returncode)


def proxy(args):
    command = [
        "python",
        "examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--prefiller-hosts",
        "127.0.0.1",
        "--prefiller-ports",
        "8100",
        "--decoder-hosts",
        "127.0.0.1",
        "--decoder-ports",
        "8200",
    ]
    logfile = args.directory / "pd-proxy.log"
    with logfile.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    state = {"pid": process.pid, "stage": "pd", "command": command, "log": str(logfile), "port": args.port}
    state_path(args).write_text(json.dumps(state, indent=2))
    record(args.directory, "pd_proxy_launch", command, state, "started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "wait", "smoke", "stop", "run", "proxy", "metrics"])
    parser.add_argument(
        "--stage", choices=["eager", "async", "graph", "mtp", "pd_prefill", "pd_decode", "pd"], default="eager"
    )
    parser.add_argument("--model", default="/mnt/weight/Hy4-preview-FP8")
    parser.add_argument("--directory", type=Path, default=Path(".vllm_ascend/hy4_adaptation"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--label", default="command")
    parser.add_argument("--command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    {
        "launch": launch,
        "wait": readiness,
        "smoke": smoke,
        "stop": stop,
        "run": run_command,
        "proxy": proxy,
        "metrics": metrics,
    }[args.action](args)


if __name__ == "__main__":
    main()
