"""Apply the version-specific Python port to an installed vLLM package.

The historical fork remains available for reproducing experiments. This
container overlay uses only its Kalypso package, never its old vLLM internals.
"""
import argparse
import ast
import shutil
from pathlib import Path


def replace_once(text, old, new, name):
    if text.count(old) != 1:
        raise RuntimeError(f"{name}: expected exactly one patch location")
    return text.replace(old, new, 1)


def prepare(target, source):
    changes = {}

    def patch(name, old, new):
        text = changes.get(name, (target / name).read_text())
        changes[name] = replace_once(text, old, new, name)

    patch("v1/request.py", "        self.request_id = request_id\n",
          "        self.request_id = request_id\n"
          "        self.kalypso_pin_key = (\n"
          "            (sampling_params.extra_args or {}).get('kalypso_pin_key')\n"
          "            if sampling_params is not None else None\n"
          "        )\n")
    patch("v1/core/sched/scheduler.py",
          "        delay_free_blocks |= connector_delay_free_blocks\n",
          "        delay_free_blocks |= connector_delay_free_blocks\n"
          "        delay_free_blocks |= bool(request.kalypso_pin_key) and (\n"
          "            request.status in (RequestStatus.FINISHED_STOPPED,\n"
          "                               RequestStatus.FINISHED_LENGTH_CAPPED)\n"
          "        )\n")
    patch("v1/engine/core.py", "    def abort_requests(self, request_ids: list[str]):\n",
          "    def unpin_request(self, request_id):\n"
          "        self.unpin_requests([request_id])\n\n"
          "    def unpin_requests(self, request_ids):\n"
          "        from vllm.kalypso.container_engine import unpin_requests\n"
          "        unpin_requests(self.scheduler, request_ids)\n\n"
          "    def get_pinned_requests(self):\n"
          "        from vllm.kalypso.container_engine import get_pinned_requests\n"
          "        return get_pinned_requests(self.scheduler)\n\n"
          "    def get_kv_cache_budget(self):\n"
          "        return self.available_gpu_memory_for_kv_cache\n\n"
          "    def get_scheduler_state(self):\n"
          "        running, waiting = self.scheduler.get_request_counts()\n"
          "        usage = self.scheduler.kv_cache_manager.usage\n"
          "        return dict(running=running, waiting=waiting,\n"
          "                    kv_cache_usage=usage,\n"
          "                    is_stuck=waiting > 0 and usage >= 0.99)\n\n"
          "    def abort_requests(self, request_ids: list[str]):\n")
    patch("entrypoints/launchers/api_server/routers.py",
          '    if "generate" in supported_tasks:\n',
          '    if "generate" in supported_tasks:\n'
          "        from vllm.kalypso.container_api import router\n"
          "        app.include_router(router)\n\n")
    patch("entrypoints/launchers/api_server/app_state.py",
          "    state.server_load_metrics = 0\n",
          "    state.server_load_metrics = 0\n"
          '    if "generate" in supported_tasks:\n'
          "        from vllm.kalypso.container_api import init_kalypso\n"
          "        await init_kalypso(engine_client, state, args)\n")

    # Validate all patch locations before modifying any upstream files.
    for name, text in changes.items():
        ast.parse(text, filename=name)
    shutil.copytree(source / "vllm/kalypso", target / "kalypso", dirs_exist_ok=True)
    overlay = source / "docker/kalypso"
    shutil.copyfile(overlay / "engine.py", target / "kalypso/container_engine.py")

    # Keep the existing semantic API, excluding the obsolete query_ref example.
    api = (source / "vllm/entrypoints/openai/api_server.py").read_text()
    routes = api[api.index("class UnpinPinnedRequestsRequest"):api.index('@router.post("/v1/semantic/query_ref")')]
    header = "import asyncio\nimport uuid\nimport pydantic\nfrom fastapi import APIRouter, Request\nfrom fastapi.responses import JSONResponse\nrouter = APIRouter()\n\n"
    init = (overlay / "init_api.py").read_text()
    changes["kalypso/container_api.py"] = header + routes + init

    imports = (
        "from vllm.entrypoints.openai.completion.protocol import CompletionRequest\n"
        "from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest\n"
    )
    for name in ("execution/vllm_executor.py", "sem_ops/endpoint.py"):
        path = target / "kalypso" / name
        text = path.read_text().replace(
            "from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest\n", imports)
        text = text.replace(
            "from vllm.entrypoints.openai.api_server import create_completion, create_chat_completion",
            "from vllm.entrypoints.openai.completion.api_router import create_completion\n"
            "from vllm.entrypoints.openai.chat_completion.api_router import create_chat_completion")
        text = text.replace("from vllm.entrypoints.openai.api_server import create_completion",
                            "from vllm.entrypoints.openai.completion.api_router import create_completion")
        if name == "execution/vllm_executor.py":
            text = "import uuid\nfrom fastapi import HTTPException\n" + text
            text = text.replace('vllm_xargs={"pinned": pin}',
                                'vllm_xargs={"kalypso_pin_key": uuid.uuid4().hex} if pin else {}')
            text = text.replace('request_id = data["id"]',
                                'request_id = req.vllm_xargs["kalypso_pin_key"] if pin else data["id"]')
            text = text.replace('        data = json.loads(raw)\n',
                                '        data = json.loads(raw)\n'
                                '        if gen.status_code >= 400:\n'
                                '            raise HTTPException(gen.status_code, detail=data)\n')
        changes["kalypso/" + name] = text
    for name, text in changes.items():
        ast.parse(text, filename=name)
        (target / name).write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.target, args.source)
