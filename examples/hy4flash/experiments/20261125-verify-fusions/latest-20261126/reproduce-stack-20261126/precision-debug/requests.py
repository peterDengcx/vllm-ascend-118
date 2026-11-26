import json
import pathlib
import time
import urllib.error
import urllib.request

r = pathlib.Path(__file__).parent
question = "请介绍一下杭州的三个主要景点，每个景点用一句话介绍。"
cases = [
    ("raw", "completions", {"prompt": "问题：" + question + "\n回答："}),
    (
        "chat",
        "chat/completions",
        {"messages": [{"role": "user", "content": question}], "chat_template_kwargs": {"reasoning_effort": "no_think"}},
    ),
    (
        "chat_math",
        "chat/completions",
        {
            "messages": [{"role": "user", "content": "17加25等于多少？只回答结果。"}],
            "chat_template_kwargs": {"reasoning_effort": "no_think"},
        },
    ),
]
results = []
for name, api, extra in cases:
    body = {"model": "hy4-flash", "max_tokens": 128, "temperature": 0, "stream": False, **extra}
    t = time.time()
    req = urllib.request.Request(
        "http://127.0.0.1:19385/v1/" + api,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as res:
            status = res.status
            response = json.load(res)
    except urllib.error.HTTPError as e:
        status = e.code
        response = e.read().decode()
    result = {"name": name, "request": body, "status": status, "elapsed": time.time() - t, "response": response}
    results.append(result)
    (r / "cases.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)
