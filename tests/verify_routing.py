"""Exercise routing against a deterministic LM Studio protocol fixture in Docker."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from fastapi.testclient import TestClient

from metarouter.config.settings import BenchmarkSettings, LMStudioSettings, Settings, set_settings
from metarouter.main import app

ROUTER = "microsoft/phi-4"
TARGET = "fixtures/coder"
REPLY = "fixture completion"
QUERY = "Show a Python function."


class Upstream(BaseHTTPRequestHandler):
    requests = []

    def do_GET(self):
        if self.path != "/api/v0/models":
            self.send_error(404)
            return
        self.send_json({"data": [{"id": model, "type": "llm", "state": "loaded"} for model in (ROUTER, TARGET)]})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append(payload)
        model = payload["model"]
        if model == ROUTER:
            content = json.dumps({"selected_model": TARGET, "reason": "coding", "confidence": 0.95, "load_required": False})
        elif model == TARGET:
            content = REPLY
        else:
            self.send_error(400, "Unexpected fixture model")
            return
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            delta = json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]})
            self.wfile.write(f"data: {delta}\n\ndata: [DONE]\n\n".encode())
        else:
            self.send_json({
                "id": "fixture-1", "object": "chat.completion", "created": 1, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            })

    def send_json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    with ThreadingHTTPServer(("127.0.0.1", 0), Upstream) as upstream:
        worker = Thread(target=upstream.serve_forever)
        worker.start()
        try:
            set_settings(Settings(
                lm_studio=LMStudioSettings(base_url=f"http://127.0.0.1:{upstream.server_port}", timeout=5),
                benchmarks=BenchmarkSettings(enabled=False),
            ))
            with TestClient(app) as client:
                assert client.get("/health").json() == {"status": "healthy"}
                assert {model["id"] for model in client.get("/v1/models").raise_for_status().json()["data"]} == {ROUTER, TARGET}
                messages = [{"role": "user", "content": QUERY}]
                request = {"model": "automatic", "messages": messages, "temperature": 0.25}
                body = client.post("/v1/chat/completions", json=request).raise_for_status().json()
                assert body["model"] == body["session_metadata"]["selected_model"] == TARGET, body
                assert body["choices"][0]["message"]["content"] == REPLY, body
                with client.stream("POST", "/v1/chat/completions", json={**request, "stream": True}) as stream:
                    stream.raise_for_status()
                    assert stream.headers["x-selected-model"] == TARGET
                    assert stream.headers["content-type"].startswith("text/event-stream")
                    events = [line.removeprefix("data: ") for line in stream.iter_lines() if line.startswith("data: ")]
                assert json.loads(events[0])["choices"][0]["delta"]["content"] == REPLY, events
                assert events[-1] == "[DONE]", events
                assert client.post("/v1/chat/completions", json={}).status_code == 422
            selection, completion, stream_selection, stream_completion = Upstream.requests
            assert [request["model"] for request in (selection, completion, stream_selection, stream_completion)] == [ROUTER, TARGET, ROUTER, TARGET]
            assert QUERY in selection["messages"][0]["content"]
            assert completion["temperature"] == stream_completion["temperature"] == 0.25
            assert completion["messages"][0]["content"] == stream_completion["messages"][0]["content"] == QUERY
            print("Verified model discovery, routed completion, SSE stream, forwarded parameters, and request validation")
        finally:
            upstream.shutdown()
            worker.join()


if __name__ == "__main__":
    main()
