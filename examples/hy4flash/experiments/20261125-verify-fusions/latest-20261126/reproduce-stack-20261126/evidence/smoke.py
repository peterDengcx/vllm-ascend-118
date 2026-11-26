import json
import pathlib
import time
import traceback
import urllib.request

r = pathlib.Path("/workspace/repro")
ports = [18100, 18200, 18201, 18202, 18203]
start = time.time()
try:
    while True:
        health = {}
        for port in ports:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as res:
                    health[str(port)] = res.status
            except Exception as e:
                health[str(port)] = str(e)
        (r / "health.json").write_text(
            json.dumps({"at": time.time(), "elapsed": time.time() - start, "ports": health}, indent=2)
        )
        if all(v == 200 for v in health.values()):
            break
        if time.time() - start > 1800:
            raise TimeoutError("Service startup exceeded 30 minutes")
        time.sleep(5)
    responses = []
    for i in range(4):
        body = {
            "model": "hy4-flash",
            "prompt": "Please explain why the sky appears blue in clear daylight. " * 40
            + f" Question {i}: Give a short explanation.",
            "max_tokens": 32,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
        }
        req = urllib.request.Request(
            "http://127.0.0.1:19385/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t = time.time()
        with urllib.request.urlopen(req, timeout=600) as res:
            status = res.status
            output = json.load(res)
        assert status == 200 and output["choices"][0]["text"] and output["usage"]["completion_tokens"] > 0, output
        responses.append({"request": body, "status": status, "elapsed": time.time() - t, "response": output})
        (r / "smoke-responses.json").write_text(json.dumps(responses, indent=2, ensure_ascii=False))
    (r / "smoke-result.json").write_text(
        json.dumps({"passed": True, "requests": len(responses), "ended": time.time()}, indent=2)
    )
except Exception:
    (r / "smoke-result.json").write_text(
        json.dumps({"passed": False, "error": traceback.format_exc(), "ended": time.time()}, indent=2)
    )
    raise
