"""Replay committed receipts with credentials removed and network connections denied."""
import os
import hashlib
import json
from pathlib import Path
import runpy
import socket
import sys
import unittest


def no_network(*args, **kwargs):
    raise RuntimeError("offline replay forbids network connections")


def main():
    root = Path(__file__).resolve().parent
    os.environ.pop("OPENAI_API_KEY",None)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(root/".work/tokenizer-cache")
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    socket.getaddrinfo = no_network
    socket.create_connection = no_network
    for entry in json.loads((root/"recorded/replay-inputs.json").read_text()):
        path = (root/entry["path"]).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("offline helper identity differs: " + entry["path"])
    sys.path.insert(0,str(root))
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover(str(root),pattern="test_*.py"))
    if not result.wasSuccessful():
        raise RuntimeError("offline contract tests failed")
    runpy.run_path(str(root/"verify.py"),run_name="__main__")


if __name__ == "__main__":
    main()
