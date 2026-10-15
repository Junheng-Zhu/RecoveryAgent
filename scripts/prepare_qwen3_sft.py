"""Prepare verified Qwen3 SFT data without a model or training dependencies."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery_agent.sft_preparation import prepare
from recovery_agent.paths import WorkspacePaths, project_path

def main():
    """Offline by default; downloading public tokenizer files is an explicit separate choice."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=WorkspacePaths.from_env().sft)
    parser.add_argument("--tokenizer", default=str(WorkspacePaths.from_env().tokenizer),
                        help="Official Hugging Face tokenizer model ID or local directory")
    parser.add_argument("--revision", default=None, help="Pin the HF tokenizer repository commit")
    parser.add_argument("--allow-tokenizer-download", action="store_true",
                        help="Allow AutoTokenizer to fetch tokenizer/config files; never weights")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--mixed", action="store_true", help="Explicit trusted-source merge with deduplication")
    args = parser.parse_args()
    workspace = WorkspacePaths.from_env()
    args.output = project_path(args.output)
    if not args.allow_tokenizer_download:
        args.tokenizer = str(project_path(args.tokenizer))
    if args.max_seq_length <= 0:
        parser.error("--max-seq-length must be positive")
    if args.allow_tokenizer_download and not args.revision:
        parser.error("--allow-tokenizer-download requires an explicit --revision")
    return prepare(ROOT, args.output, args.tokenizer, args.max_seq_length,
                   args.audit_only, not args.allow_tokenizer_download, args.revision, args.mixed)

if __name__ == "__main__":
    raise SystemExit(main())
