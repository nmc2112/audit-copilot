"""Command line: python -m audit_copilot.cli --fec data/sample_fec.txt "question"."""

from __future__ import annotations

import argparse
import logging

from .agent import AuditAgent
from .config import load_settings
from .fec import load_fec
from .llm import build_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the audit agent a question about a FEC file")
    parser.add_argument("question")
    parser.add_argument("--fec", default="data/sample_fec.txt")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    settings = load_settings()
    result = AuditAgent(load_fec(args.fec), settings, build_client(settings.llm)).run(args.question)
    print(result.report.model_dump_json(indent=2))
    print(f"\nsource={result.report_source} iterations={result.iterations} "
          f"tokens={result.prompt_tokens}+{result.completion_tokens} latency={result.latency_s}s")


if __name__ == "__main__":
    main()
