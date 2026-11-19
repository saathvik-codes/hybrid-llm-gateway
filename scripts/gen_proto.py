"""Generate the gRPC Python stubs from proto/gateway.proto into app/ (they are git-ignored: build output).

    python scripts/gen_proto.py
"""
import re
import sys
from pathlib import Path

from grpc_tools import protoc

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    code = protoc.main(["protoc", f"-I{ROOT / 'proto'}", f"--python_out={ROOT / 'app'}",
                        f"--grpc_python_out={ROOT / 'app'}", str(ROOT / "proto" / "gateway.proto")])
    grpc_file = ROOT / "app" / "gateway_pb2_grpc.py"
    if code == 0 and grpc_file.exists():       # protoc writes `import gateway_pb2`; inside a package it must be relative
        grpc_file.write_text(re.sub(r"^import gateway_pb2 as", "from app import gateway_pb2 as",
                                    grpc_file.read_text(encoding="utf-8"), flags=re.M), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
