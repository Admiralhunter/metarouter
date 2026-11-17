"""Exercise routing against a deterministic LM Studio protocol fixture in Docker."""

import asyncio
import json
from contextlib import chdir, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from unittest.mock import patch

import yaml
from fastapi.testclient import TestClient

from metarouter.api.routes import CompletionStream
from metarouter.config.settings import (
    BenchmarkSettings, LMStudioSettings, Settings, get_settings, set_settings,
)
from metarouter.main import app
from metarouter.routing.router import ModelRouter

ROUTER = "microsoft/phi-4"
TARGET = "fixtures/coder"
REPLY = "fixture completion"
QUERY = "Show a Python function."


class Upstream(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/api/v0/models":
            self.send_error(404)
            return
        if not self.server.available:
            self.send_error(503)
            return
        self.send_json({"data": [
            {"id": model, "type": "llm", "state": state}
            for model, state in self.server.models.items()
        ]})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(payload)
        model = payload["model"]
        if not self.server.available or model not in self.server.models:
            self.send_error(503, "Model unavailable on this fixture")
            return
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


@contextmanager
def upstream_server(models):
    with ThreadingHTTPServer(("127.0.0.1", 0), Upstream) as server:
        server.models = models
        server.available = True
        server.requests = []
        server.url = f"http://127.0.0.1:{server.server_port}"
        worker = Thread(target=server.serve_forever)
        worker.start()
        try:
            yield server
        finally:
            server.shutdown()
            worker.join()


def verify_single_instance():
    with upstream_server({ROUTER: "loaded", TARGET: "loaded"}) as upstream:
        set_settings(Settings(
            lm_studio=LMStudioSettings(base_url=upstream.url, timeout=5),
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
        selection, completion, stream_selection, stream_completion = upstream.requests
        assert [request["model"] for request in (selection, completion, stream_selection, stream_completion)] == [ROUTER, TARGET, ROUTER, TARGET]
        assert QUERY in selection["messages"][0]["content"]
        assert completion["temperature"] == stream_completion["temperature"] == 0.25
        assert completion["messages"][0]["content"] == stream_completion["messages"][0]["content"] == QUERY
        print("Verified model discovery, routed completion, SSE stream, forwarded parameters, and request validation")


async def verify_stream_reservations(settings):
    client = ModelRouter(settings).client
    await client.get_models()
    messages = [{"role": "user", "content": QUERY}]
    never_started = await client.chat_completion(TARGET, messages, stream=True)
    await never_started.aclose()
    assert all(row["current_load"] == 0 for row in (await client.get_instance_health()).values())

    stream = await client.chat_completion(TARGET, messages, stream=True)
    try:
        assert REPLY.encode() in await anext(stream)
        assert (await client.get_instance_health())["primary"]["current_load"] > 0
        await client.chat_completion(TARGET, messages)
    finally:
        await stream.aclose()
    assert all(row["current_load"] == 0 for row in (await client.get_instance_health()).values())

    async def disconnected_send(message):
        if message["type"] == "http.response.body":
            raise OSError("Fixture client disconnected")

    response = CompletionStream(await client.chat_completion(TARGET, messages, stream=True))
    try:
        await response.stream_response(disconnected_send)
    except OSError:
        pass
    else:
        raise AssertionError("Fixture did not simulate a client disconnection")
    assert all(row["current_load"] == 0 for row in (await client.get_instance_health()).values())


def verify_multiple_instances():
    with (
        upstream_server({ROUTER: "loaded", TARGET: "not-loaded"}) as primary,
        upstream_server({TARGET: "loaded"}) as secondary,
        TemporaryDirectory() as temp,
        chdir(temp),
    ):
        config = Path("config/router.yaml")
        config.parent.mkdir()
        config.write_text(yaml.safe_dump({
            "lm_studio": {"instances": [
                {"name": "primary", "base_url": primary.url},
                {"name": "secondary", "base_url": secondary.url},
            ]},
            "benchmarks": {"enabled": False},
        }), encoding="utf-8")
        settings = get_settings()
        assert len(settings.lm_studio.instances) == 2
        with patch.dict("os.environ", {"LMSTUDIO_BASE_URL": secondary.url}):
            assert LMStudioSettings(base_url=primary.url).base_url == secondary.url
        request = {"messages": [{"role": "user", "content": QUERY}]}

        with TestClient(app) as api:
            models = api.get("/v1/models").raise_for_status().json()["data"]
            assert {model["id"]: model["instance_name"] for model in models} == {
                ROUTER: "primary", TARGET: "secondary",
            }
            assert api.get("/v1/instances").raise_for_status().json()["healthy_instances"] == 2
            assert api.post("/v1/chat/completions", json=request).raise_for_status().json()["model"] == TARGET
            assert [payload["model"] for payload in primary.requests] == [ROUTER]
            assert [payload["model"] for payload in secondary.requests] == [TARGET]
            stream = api.post("/v1/chat/completions", json={**request, "stream": True}).raise_for_status()
            assert REPLY in stream.text and stream.text.rstrip().endswith("data: [DONE]")

            secondary.available = False
            health = api.get("/v1/instances").raise_for_status().json()
            assert health["healthy_instances"] == 1
            assert health["instances"]["secondary"]["status"] == "unreachable"
            assert api.post("/v1/chat/completions", json=request).raise_for_status().json()["model"] == TARGET
            assert primary.requests[-1]["model"] == TARGET
            secondary.available = True
            assert api.get("/v1/instances").raise_for_status().json()["healthy_instances"] == 2
            assert api.post("/v1/chat/completions", json=request).raise_for_status().json()["model"] == TARGET
            assert secondary.requests[-1]["model"] == TARGET

        primary.models[TARGET] = "loaded"
        calls = (len(primary.requests), len(secondary.requests))
        asyncio.run(verify_stream_reservations(settings))
        assert len(primary.requests) - calls[0] == 2
        assert len(secondary.requests) - calls[1] == 1
        print("Verified YAML discovery, replica selection, outage recovery, balancing, and stream cleanup")


if __name__ == "__main__":
    verify_multiple_instances()
    verify_single_instance()
