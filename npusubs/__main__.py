import argparse
import logging
import os

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(prog="npusubs", description="Subtitle-grade Whisper on the AMD XDNA1 NPU")
    ap.add_argument("--host", default=os.environ.get("NPUSUBS_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("NPUSUBS_PORT", "9091")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run("npusubs.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
