"""Build / refresh the persistent ChromaDB index.
Usage: python scripts/build_index.py [--limit N] [--rebuild]"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ingestion import build_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None, help="index only the first N records (development subset)")
ap.add_argument("--rebuild", action="store_true", help="drop the collection and rebuild from scratch")
args = ap.parse_args()
print(build_index(limit=args.limit, rebuild=args.rebuild).to_text())
