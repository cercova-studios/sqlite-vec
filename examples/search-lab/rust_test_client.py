"""Test-only JSONL client for the Rust binary; no retrieval implementation."""

import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
BINARY=HERE/"target/release/sqlite-vec-search-lab"


def command(config,transport="cli"):
    return [str(BINARY),"--config",str(config),transport]


class RustAdapter:
    def __init__(self,config):
        self.process=subprocess.Popen(command(config),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    def call(self,record):
        self.process.stdin.write(json.dumps(record,separators=(",",":"),ensure_ascii=True).encode()+b"\n")
        self.process.stdin.flush()
        line=self.process.stdout.readline()
        assert line, self.process.stderr.read().decode()
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        try:
            assert self.process.wait(timeout=10) in (0,1)
            assert not self.process.stderr.read()
        finally:
            if self.process.poll() is None:
                self.process.kill();self.process.wait()
            self.process.stdout.close();self.process.stderr.close()
