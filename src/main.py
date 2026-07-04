"""CLI entry point for the Anki deck generator pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.clean import run_clean
from src.config import CLASSIFY_TOP_N, DEFAULT_TOP_N, TRANSLATE_TOP_N
from src.export import run_export
from src.filter import run_filter
from src.llm import run_generate
from src.rank import run_rank
from src.translate import run_translate

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_prepare_data(_args: argparse.Namespace) -> None:
    """Run reference data preparation script."""
    import subprocess

    script = Path(__file__).resolve().parent.parent / "scripts" / "prepare_reference_data.py"
    subprocess.run([sys.executable, str(script)], check=True)


def cmd_clean(args: argparse.Namespace) -> None:
    run_clean(input_path=args.input, output_path=args.output)


def cmd_classify(args: argparse.Namespace) -> None:
    from src.classify import run_classify
    from src.seed_labels import get_seed_labels

    run_classify(
        input_path=args.input,
        top_n=args.top,
        dry_run=args.dry_run,
        use_cache=not args.no_cache,
        refresh_lemma=args.refresh_lemma,
        seed_labels=get_seed_labels(),
    )


def cmd_translate(args: argparse.Namespace) -> None:
    from src.seed_dict import get_seed_dict

    run_translate(
        input_path=args.input,
        top_n=args.top,
        dry_run=args.dry_run,
        use_cache=not args.no_cache,
        refresh_lemma=args.refresh_lemma,
        seed_dict=get_seed_dict(),
    )


def cmd_filter(args: argparse.Namespace) -> None:
    run_filter(input_path=args.input, output_path=args.output)


def cmd_rank(args: argparse.Namespace) -> None:
    run_rank(input_path=args.input, output_path=args.output, top_n=args.top)


def cmd_generate(args: argparse.Namespace) -> None:
    run_generate(
        input_path=args.input,
        top_n=args.top,
        dry_run=args.dry_run,
        use_cache=not args.no_cache,
        refresh_lemma=args.refresh_lemma,
        output_path=args.output,
        model=args.model,
        force=args.force,
    )


def cmd_export(args: argparse.Namespace) -> None:
    run_export(
        input_path=args.input,
        output_path=args.output,
        excel_bom=args.excel_bom,
        apkg=args.apkg,
        apkg_path=args.apkg_output,
        with_audio=not args.no_audio,
    )


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run full pipeline."""
    cmd_prepare_data(args)
    run_clean()

    if not args.skip_classify:
        cmd_classify(args)

    if not args.skip_translate:
        cmd_translate(args)

    run_filter()
    run_rank(top_n=args.top)

    if not args.skip_generate:
        run_generate(
            top_n=args.top,
            dry_run=args.dry_run,
            use_cache=not args.no_cache,
        )
        if not args.dry_run:
            run_export(excel_bom=args.excel_bom)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Brazilian Portuguese Anki deck generator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data", help="Write seed PT→ES dictionary")

    p_clean = sub.add_parser("clean", help="Clean frequency list")
    p_clean.add_argument("--input", help="Input TSV path")
    p_clean.add_argument("--output", help="Output parquet path")
    p_clean.set_defaults(func=cmd_clean)

    p_classify = sub.add_parser("classify", help="LLM studyability classification")
    p_classify.add_argument("--input", help="Input cleaned parquet")
    p_classify.add_argument("--top", type=int, default=CLASSIFY_TOP_N)
    p_classify.add_argument("--dry-run", action="store_true")
    p_classify.add_argument("--no-cache", action="store_true")
    p_classify.add_argument("--refresh-lemma", help="Reclassify single lemma")
    p_classify.set_defaults(func=cmd_classify)

    p_translate = sub.add_parser("translate", help="Build PT→ES translation dict via LLM")
    p_translate.add_argument("--input", help="Input cleaned parquet")
    p_translate.add_argument("--top", type=int, default=TRANSLATE_TOP_N)
    p_translate.add_argument("--dry-run", action="store_true")
    p_translate.add_argument("--no-cache", action="store_true")
    p_translate.add_argument("--refresh-lemma", help="Retranslate single lemma")
    p_translate.set_defaults(func=cmd_translate)

    p_filter = sub.add_parser("filter", help="Apply translation-first cognate filtering")
    p_filter.add_argument("--input", help="Input cleaned parquet")
    p_filter.add_argument("--output", help="Output scored parquet")
    p_filter.set_defaults(func=cmd_filter)

    p_rank = sub.add_parser("rank", help="Rank by priority score")
    p_rank.add_argument("--input", help="Input scored parquet")
    p_rank.add_argument("--output", help="Output ranked parquet")
    p_rank.add_argument("--top", type=int, help="Keep top N rows")
    p_rank.set_defaults(func=cmd_rank)

    p_gen = sub.add_parser("generate", help="Generate LLM cards")
    p_gen.add_argument("--input", help="Input ranked parquet")
    p_gen.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    p_gen.add_argument("--dry-run", action="store_true")
    p_gen.add_argument("--no-cache", action="store_true")
    p_gen.add_argument("--refresh-lemma", help="Regenerate single lemma")
    p_gen.add_argument("--model", help="OpenRouter model id (e.g. anthropic/claude-sonnet-4.6)")
    p_gen.add_argument("--output", help="Output cards.jsonl path")
    p_gen.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    p_gen.set_defaults(func=cmd_generate)

    p_export = sub.add_parser("export", help="Export Anki CSV and/or .apkg")
    p_export.add_argument("--input", help="Input cards.jsonl")
    p_export.add_argument("--output", help="Output CSV path")
    p_export.add_argument("--excel-bom", action="store_true")
    p_export.add_argument("--apkg", action="store_true", help="Also write ready-to-import .apkg")
    p_export.add_argument("--apkg-output", help="Output .apkg path")
    p_export.add_argument("--no-audio", action="store_true", help="Skip Edge TTS when building .apkg")
    p_export.set_defaults(func=cmd_export)

    p_all = sub.add_parser("run-all", help="Run full pipeline")
    p_all.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    p_all.add_argument("--dry-run", action="store_true")
    p_all.add_argument("--no-cache", action="store_true")
    p_all.add_argument("--skip-translate", action="store_true")
    p_all.add_argument("--skip-classify", action="store_true")
    p_all.add_argument("--skip-generate", action="store_true")
    p_all.add_argument("--excel-bom", action="store_true")
    p_all.set_defaults(func=cmd_run_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
