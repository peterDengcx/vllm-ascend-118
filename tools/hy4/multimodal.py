# SPDX-License-Identifier: Apache-2.0
"""Actual image/video curl probes and encoder/fusion evidence for random HY4 vision."""

import base64
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from validate import record


def create_assets(directory):
    assets = directory / "media"
    assets.mkdir(exist_ok=True)
    for name, color in (("red", "red"), ("blue", "blue")):
        Image.new("RGB", (64, 64), color).save(assets / f"{name}.png")
    checker = Image.new("RGB", (128, 64), "white")
    draw = ImageDraw.Draw(checker)
    for row in range(4):
        for column in range(8):
            if (row + column) % 2:
                draw.rectangle((column * 16, row * 16, column * 16 + 15, row * 16 + 15), fill="black")
    checker.save(assets / "checker.png")
    path = assets / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2, (64, 64))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot encode the HY4 test video")
    for color in ("red", "blue", "green", "white"):
        frame = np.asarray(Image.new("RGB", (64, 64), color))[:, :, ::-1].copy()
        writer.write(frame)
    writer.release()
    return {name: assets / name for name in ("red.png", "blue.png", "checker.png", "clip.mp4")}


def data_url(path):
    mime = "video/mp4" if path.suffix == ".mp4" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def smoke_multimodal(args, state):
    assets = create_assets(args.directory)

    def image(name):
        return {"type": "image_url", "image_url": {"url": data_url(assets[name])}}

    def text(prompt):
        return {"type": "text", "text": prompt}

    tests = [
        ("single_red", [image("red.png"), text("忽略图片内容，只用中文回答：测试完成。")]),
        ("single_blue", [image("blue.png"), text("忽略图片内容，只用中文回答：测试完成。")]),
        (
            "non_square",
            [image("checker.png"), text("This is an integration test. Reply with one short English sentence.")],
        ),
        (
            "two_images",
            [
                image("red.png"),
                text("这里有两张测试图片。"),
                image("blue.png"),
                text("无需识别内容，只回复一句简短中文。"),
            ],
        ),
        (
            "video",
            [
                {"type": "video_url", "video_url": {"url": data_url(assets["clip.mp4"])}},
                text("这是测试视频。无需识别画面，回复一句简短中文。"),
            ],
        ),
        (
            "image_long_context",
            [image("checker.png"), text("这是用于检查长上下文的填充文字。" * 300 + "忽略图片，只回答1+1等于几。")],
        ),
    ]
    failures = 0
    for name, content in tests:
        body = {
            "model": "hy4",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 64,
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
            answer = decoded["choices"][0]["message"]["content"] or ""
            passed = (
                code == "200"
                and bool(answer.strip())
                and "\ufffd" not in answer
                and any(c.isalpha() or c.isdigit() for c in answer)
            )
        except (ValueError, KeyError, IndexError):
            pass
        record(
            args.directory,
            state["stage"] + "_multimodal_curl",
            {"case": name, "request": body, "semantic_accuracy_evaluated": False},
            {"http_code": code, "response": response, "stderr": result.stderr, "seconds": time.monotonic() - started},
            "passed" if passed else "failed",
        )
        failures += not passed
    instances = [state]
    if state["stage"] == "pd":
        instances = [json.loads((args.directory / f"{name}.json").read_text()) for name in ("pd_prefill", "pd_decode")]
    evidence = []
    for instance in instances:
        raw = Path(instance["log"]).read_bytes()[instance.get("ready_log_offset", 0) :].decode(errors="replace")
        evidence += [f"{instance['stage']}: {line}" for line in raw.splitlines() if "HY4_VL_" in line]
    encoders = [line for line in evidence if "HY4_VL_ENCODER" in line]
    fusions = [line for line in evidence if "HY4_VL_FUSION" in line]
    checksums = {float(match.group(1)) for line in encoders if (match := re.search(r"sum=([-\d.]+)", line))}
    finite = bool(encoders) and all("finite=True" in line for line in encoders)
    fused = any(int(match.group(1)) > 0 for line in fusions if (match := re.search(r"visual_tokens=(\d+)", line)))
    passed = finite and fused and len(checksums) > 1
    record(
        args.directory,
        state["stage"] + "_multimodal_evidence",
        {
            "assets": {
                name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
                for name, path in assets.items()
            }
        },
        {
            "encoder_calls": len(encoders),
            "fusion_calls": len(fusions),
            "all_finite": finite,
            "image_dependent_embeddings": len(checksums) > 1,
            "runtime_log": evidence,
        },
        "passed" if passed else "failed",
    )
    return failures + (not passed)
