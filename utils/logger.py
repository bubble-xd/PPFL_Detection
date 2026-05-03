from __future__ import annotations

import os
import traceback
from datetime import datetime


class RunLogger:
    def __init__(self, log_path: str, print_to_console: bool = True) -> None:
        self.log_path = log_path
        self.print_to_console = print_to_console
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write("")

    def _format(self, level: str, message: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{level}] {message}"

    def log(self, level: str, message: str) -> None:
        line = self._format(level=level, message=message)
        if self.print_to_console:
            print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def info(self, message: str) -> None:
        self.log("INFO", message)

    def error(self, message: str) -> None:
        self.log("ERROR", message)

    def exception(self, message: str) -> None:
        self.error(message)
        stack = traceback.format_exc().rstrip()
        if stack and stack != "NoneType: None":
            for line in stack.splitlines():
                self.error(line)
