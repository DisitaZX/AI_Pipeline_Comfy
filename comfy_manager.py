"""
ComfyManager: запуск/остановка/рестарт локального ComfyUI как subprocess.

Используется как барьер между этапами пайплайна (image -> video и т.п.),
чтобы полностью освобождать ОЗУ и VRAM, а не полагаться на cleanup-ноды
внутри workflow'а (которые между задачами одного этапа только мешают).

Особенности:
  - Жёсткий kill всего дерева процессов на Windows (taskkill /F /T).
  - Ожидание освобождения порта (TIME_WAIT после kill).
  - Ожидание готовности через GET /system_stats.
  - Логирование stdout/stderr Comfy в файл - чтобы при ночном запуске
    можно было разобрать что упало.
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


class ComfyManager:
    def __init__(
            self,
            comfy_root: str | os.PathLike,
            host: str = "127.0.0.1",
            port: int = 8188,
            bat_name: str = "run_nvidia_gpu.bat",
            launch_cmd: Sequence[str] | None = None,
            log_path: str | os.PathLike | None = None,
            startup_timeout: float = 240.0,
            shutdown_timeout: float = 30.0,
            port_free_timeout: float = 60.0,
    ):
        """
        comfy_root: корень ComfyUI_windows_portable (содержит run_nvidia_gpu.bat).
        bat_name:   имя bat-файла запуска (по умолчанию стандартный portable-launcher).
        launch_cmd: опционально - своя команда запуска целиком. Если задана -
                    bat_name игнорируется. Используй если запускаешь Comfy
                    из venv/conda/Linux.
        host/port:  на ЧТО проверять готовность. Должны совпадать с тем, что
                    реально слушает Comfy после .bat. Стандартный bat поднимает
                    127.0.0.1:8188 - менять не нужно. Если меняешь - правь сам .bat.
        """
        self.comfy_root = Path(comfy_root).resolve()
        self.host = host
        self.port = port

        if launch_cmd is None:
            bat_path = self.comfy_root / bat_name
            if sys.platform == "win32":
                # cmd.exe /c <bat>: bat запускается, taskkill /T снесёт всё дерево
                # cmd -> python_embeded.exe -> воркеры.
                launch_cmd = ["cmd.exe", "/c", str(bat_path)]
            else:
                # На Linux .bat-файлов не бывает - оставляем явное падение,
                # пусть пользователь передаст свой launch_cmd.
                raise RuntimeError(
                    "Default launch_cmd requires Windows. "
                    "On Linux/Mac pass `launch_cmd=[...]` explicitly."
                )
        self.launch_cmd = list(launch_cmd)

        if log_path is None:
            log_path = self.comfy_root / "comfy_manager.log"
        self.log_path = Path(log_path)

        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.port_free_timeout = port_free_timeout
        self._proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

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
        deadline = time.monotonic() + self.startup_timeout
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while time.monotonic() < deadline:
                # Если процесс уже умер на старте - не ждём впустую.
                if self._proc and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"ComfyUI process exited with code {self._proc.returncode} "
                        f"during startup. See log: {self.log_path}"
                    )
                try:
                    async with session.get(f"{self.base_url}/system_stats") as r:
                        if r.status == 200:
                            await r.read()
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(1.0)
        raise TimeoutError(
            f"ComfyUI did not become ready within {self.startup_timeout}s. "
            f"See log: {self.log_path}"
        )

    # ---------------------------------------------------------------- public

    async def start(self) -> None:
        """Запускает Comfy и ждёт пока он не ответит на /system_stats."""
        if self._proc and self._proc.poll() is None:
            return  # уже работает
        if self._is_port_in_use():
            # Кто-то держит порт. Не запускаем второй экземпляр поверх -
            # ждём, либо падаем по таймауту.
            await self._wait_port_free()

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(self.log_path, "ab", buffering=0)
        log_f.write(
            f"\n\n=== ComfyUI start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode()
        )

        creationflags = 0
        if sys.platform == "win32":
            # CREATE_NEW_PROCESS_GROUP позволяет посылать сигналы отдельно от родителя
            # и упрощает taskkill /T (kill всего дерева).
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self._proc = subprocess.Popen(
            self.launch_cmd,
            cwd=str(self.comfy_root),
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
        """Жёстко убивает дерево процессов Comfy и ждёт освобождения порта."""
        if not self._proc or self._proc.poll() is not None:
            self._proc = None
            # Даже если своего процесса нет - всё равно ждём порт,
            # вдруг там зомби от прошлой сессии.
            await self._wait_port_free()
            return

        pid = self._proc.pid

        if sys.platform == "win32":
            # taskkill /T /F убивает весь tree - Comfy форкает дочерние воркеры,
            # обычный terminate их не достанет.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            self._proc.terminate()

        # Ждём пока процесс действительно умрёт.
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

    async def restart(self) -> None:
        """Полный цикл: stop -> ждём освобождения порта -> start -> ждём ready."""
        await self.stop()
        await self.start()


COMFY_ROOT = r"C:\Users\Loopy\Desktop\ComfyUI_windows_portable"

comfy_mgr = ComfyManager(
    comfy_root=COMFY_ROOT,
    host="127.0.0.1",
    port=8188,
    # extra_args=["--lowvram"],  # если нужно
)

def main():
    asyncio.run(comfy_mgr.start())
    time.sleep(10)
    asyncio.run(comfy_mgr.restart())
    time.sleep(10)
    asyncio.run(comfy_mgr.stop())
if __name__ == "__main__":
    main()
