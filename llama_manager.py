"""
LlamaServerManager: запуск/остановка локального llama.cpp llama-server.

Используется как барьер вокруг LLM-этапа пайплайна (получение video-prompt'ов
от Qwen3.6-35B-A3B). До этапа — start(), модель загружается в RAM/VRAM.
После этапа — stop(), вся память возвращается ОС, ComfyUI получает её
обратно для генерации видео.

Особенности:
  - Жёсткий kill всего дерева на Windows (taskkill /F /T) — иначе
    дочерний llama-server.exe пережил бы родительский cmd.exe.
  - Опциональный --mmproj для multimodal моделей (Qwen3-VL и т.п.).
  - Health-check через GET /health (поддерживается llama.cpp с лета 2024).
  - Логирование stdout/stderr llama-server'а в файл — чтобы при падении
    модели было понятно почему (OOM, плохой mmproj, и т.п.).
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import aiohttp


class LlamaServerManager:
    def __init__(
        self,
        llama_server_bin: str | os.PathLike,
        model_path: str | os.PathLike,
        host: str = "127.0.0.1",
        port: int = 8080,
        mmproj_path: str | os.PathLike | None = None,
        ctx_size: int = 65536,
        n_cpu_moe: int | None = 33,
        no_mmap: bool = True,
        mlock: bool = True,
        cache_type_k: str = "turbo4",
        cache_type_v: str = "turbo3",
        extra_args: Sequence[str] = (),
        log_path: str | os.PathLike | None = None,
        startup_timeout: float = 600.0,   # 35B-A3B на CPU грузится ~3-5 мин
        shutdown_timeout: float = 30.0,
        port_free_timeout: float = 60.0,
    ):
        """
        Параметры в основном повторяют CLI llama-server.

        llama_server_bin: путь к llama-server.exe (или просто 'llama-server'
                          если в PATH).
        model_path:       путь к .gguf модели.
        mmproj_path:      путь к mmproj-*.gguf (None = text-only режим).
        n_cpu_moe:        количество MoE-экспертов на CPU. None = не передавать.
        extra_args:       прочие CLI-аргументы вперемешку (например ["--n-gpu-layers", "99"]).
        startup_timeout:  сколько ждать готовности после старта. 35B на
                          CPU-выгрузке грузится медленно — оставлен с запасом.
        """
        self.llama_server_bin = str(llama_server_bin)
        self.model_path = str(model_path)
        self.mmproj_path = str(mmproj_path) if mmproj_path else None
        self.host = host
        self.port = port

        cmd: list[str] = [
            self.llama_server_bin,
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(ctx_size),
            "--cache-type-k", cache_type_k,
            "--cache-type-v", cache_type_v,
        ]
        if n_cpu_moe is not None:
            cmd += ["--n-cpu-moe", str(n_cpu_moe)]
        if no_mmap:
            cmd.append("--no-mmap")
        if mlock:
            cmd.append("--mlock")
        if self.mmproj_path:
            cmd += ["--mmproj", self.mmproj_path]
        cmd.extend(extra_args)
        self.launch_cmd = cmd

        if log_path is None:
            log_path = Path(self.model_path).parent / "llama_server.log"
        self.log_path = Path(log_path)

        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.port_free_timeout = port_free_timeout
        self._proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    # ------------------------------------------------------------------ utils

    def _is_port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex((self.host, self.port)) == 0

    async def _wait_port_free(self) -> None:
        deadline = time.monotonic() + self.port_free_timeout
        while time.monotonic() < deadline:
            if not self._is_port_in_use():
                return
            await asyncio.sleep(0.5)
        raise TimeoutError(
            f"Port {self.port} still in use after {self.port_free_timeout}s"
        )

    async def _wait_ready(self) -> None:
        """
        Ждёт пока /health не вернёт 200. llama-server во время загрузки
        модели отдаёт 503 'loading model' — мы это терпим и продолжаем
        опрашивать. По истечении timeout'а — TimeoutError.
        """
        deadline = time.monotonic() + self.startup_timeout
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while time.monotonic() < deadline:
                if self._proc and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited with code {self._proc.returncode} "
                        f"during startup. See log: {self.log_path}"
                    )
                try:
                    async with session.get(f"{self.base_url}/health") as r:
                        if r.status == 200:
                            await r.read()
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(2.0)
        raise TimeoutError(
            f"llama-server did not become ready within {self.startup_timeout}s. "
            f"See log: {self.log_path}"
        )

    # ---------------------------------------------------------------- public

    async def start(self) -> None:
        """Запускает llama-server и ждёт пока /health не отдаст 200."""
        if self._proc and self._proc.poll() is None:
            return
        if self._is_port_in_use():
            await self._wait_port_free()

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(self.log_path, "ab", buffering=0)
        log_f.write(
            f"\n\n=== llama-server start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode()
        )
        log_f.write(f"cmd: {' '.join(self.launch_cmd)}\n".encode())

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self._proc = subprocess.Popen(
            self.launch_cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            close_fds=False,
        )
        try:
            await self._wait_ready()
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Жёстко гасит llama-server и ждёт освобождения порта."""
        if not self._proc or self._proc.poll() is not None:
            self._proc = None
            await self._wait_port_free()
            return

        pid = self._proc.pid

        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            self._proc.terminate()

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: self._proc.wait(timeout=self.shutdown_timeout)
            )
        except subprocess.TimeoutExpired:
            self._proc.kill()
            try:
                await loop.run_in_executor(None, self._proc.wait, 5)
            except Exception:
                pass

        self._proc = None
        await self._wait_port_free()


# ---------------------------------------------------------------- singleton

LLAMA_SERVER_BIN = r"C:\Users\Loopy\Desktop\llama_TurboQuant\llama-server.exe"
LLAMA_MODEL = r"C:\local_models\gemma-4-E4B-it-UD-Q8_K_XL.gguf"
LLAMA_MMPROJ = r"C:\local_models\gemma-4-E4B-it-mmproj\mmproj-BF16.gguf"

llama_mgr = LlamaServerManager(
    llama_server_bin=LLAMA_SERVER_BIN,
    model_path=LLAMA_MODEL,
    mmproj_path=LLAMA_MMPROJ,
    host="127.0.0.1",
    port=8080,
    n_cpu_moe=None,  # Gemma не MoE
    extra_args=["--n-gpu-layers", "999"],  # full GPU offload
    ctx_size=65536,
    no_mmap=True,
    mlock=True,
    cache_type_k="f16",  # дефолт; отключаем turbo3/4
    cache_type_v="f16",  # они для MoE-cache не нужны на dense
)


def main():
    """Sanity-check: запустить, дождаться готовности, сразу погасить."""
    async def _run():
        await llama_mgr.start()
        print(f"OK, listening on {llama_mgr.base_url}")
        await asyncio.sleep(2)
        await llama_mgr.stop()
        print("stopped")
    asyncio.run(_run())


if __name__ == "__main__":
    main()