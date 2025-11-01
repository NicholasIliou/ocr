#!/usr/bin/env python3
"""
Multi-language OCR Processor

Reads images from a folder, runs Tesseract OCR (eng+deu by default),
optionally translates output, heuristically preserves table-like layout,
and writes JSON/CSV results.

Usage (example):
    python ocr_processor.py --input_dir ./images --output_json ocr_results.json --translate --dest_lang en

Requirements:
    - Python 3.9+
    - Tesseract OCR engine installed separately and available on PATH
    - pip install -r requirements.txt

"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from PIL import Image, ImageFile

# allow truncated images to open where possible
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import pytesseract
    from pytesseract import Output
except Exception:
    pytesseract = None  # will raise at runtime if used without install

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

# optional progress bar dependency; prefer tqdm if installed
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def find_image_files(input_dir: str) -> List[str]:
    files = []
    for root, _dirs, filenames in os.walk(input_dir):
        for fn in filenames:
            _, ext = os.path.splitext(fn)
            if ext.lower() in IMAGE_EXTENSIONS:
                files.append(os.path.join(root, fn))
    return sorted(files)


def heuristic_preserve_table(image: Image.Image, text: str) -> Tuple[str, bool]:
    """
    Try to improve preservation of table-like content.

    Strategy (simple heuristic):
    - Use pytesseract.image_to_data to get word positions.
    - Group words into lines by y-position.
    - Within each line, if gaps between adjacent words exceed a threshold,
      insert a tab between them. This often helps keep columns aligned.

    Returns the (possibly reformatted) text and a boolean indicating whether
    reformatting looked like table-like output.
    """
    if pytesseract is None:
        return text, False

    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
    except Exception:
        return text, False

    n_boxes = len(data.get("text", []))
    if n_boxes == 0:
        return text, False

    # collect words with coordinates
    rows: Dict[int, List[Tuple[int, int, str]]] = {}
    for i in range(n_boxes):
        word = data["text"][i]
        if not word or word.strip() == "":
            continue
        x = int(data.get("left", [0])[i])
        y = int(data.get("top", [0])[i])
        # bucket y to rows using tolerance
        row_key = round(y / 10)  # 10-pixel granularity
        rows.setdefault(row_key, []).append((x, y, word))

    if not rows:
        return text, False

    # build lines sorted by y
    lines = []
    for row_key in sorted(rows.keys()):
        items = sorted(rows[row_key], key=lambda t: t[0])
        xs = [it[0] for it in items]
        # compute dynamic gap threshold from median char width estimate
        gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
        median_gap = int(sorted(gaps)[len(gaps) // 2]) if gaps else 0
        # threshold: gaps significantly bigger than median -> treat as column gap
        threshold = max(20, int(median_gap * 1.8))

        parts = []
        last_x = None
        for x, _y, w in items:
            if last_x is None:
                parts.append(w)
            else:
                if x - last_x > threshold:
                    parts.append("\t" + w)
                else:
                    parts.append(" " + w)
            last_x = x + len(w) * 6  # crude next position estimate

        # join and normalize: collapse multiple spaces
        line = "".join(parts)
        line = re.sub(r"\s+", " ", line).replace("\t ", "\t")
        lines.append(line.strip())

    # decide if this is table-like: if many lines contain tabs or consistent column counts
    tab_lines = sum(1 for ln in lines if "\t" in ln)
    is_table_like = tab_lines >= max(1, len(lines) // 6)

    if is_table_like:
        return "\n".join(lines), True
    return text, False


def ocr_image(path: str, languages: str = "eng+deu") -> Optional[str]:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed or not importable")
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            # Let tesseract try to preserve layout; we'll also run our heuristic after
            text = pytesseract.image_to_string(img, lang=languages)
            return text
    except Exception as e:
        logging.warning("Failed to OCR %s: %s", path, e)
        return None


def translate_text(text: str, target_lang: str = "en") -> Optional[str]:
    if GoogleTranslator is None:
        logging.warning("Translation requested but deep-translator is not installed")
        return None
    try:
        translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
        return translated
    except Exception as e:
        logging.warning("Translation failed: %s", e)
        return None


def process_folder(
    input_dir: str,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    languages: str = "eng+deu",
    translate: bool = False,
    dest_lang: str = "en",
    parallel: bool = True,
    workers: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Dict[str, Optional[str]]]:
    results: Dict[str, Dict[str, Optional[str]]] = {}
    image_files = find_image_files(input_dir)
    if not image_files:
        logging.warning("No image files found in %s", input_dir)
        return results
    logging.info("Found %d image(s) to process", len(image_files))

    # helper to process a single image (safe to run in threads)
    def _process_one(img_path: str) -> Optional[Tuple[str, Dict[str, Optional[str]]]]:
        name = os.path.basename(img_path)
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if pytesseract is None:
                    raise RuntimeError("pytesseract is required but not available")

                original_text = pytesseract.image_to_string(img, lang=languages)
                formatted_text, was_table = heuristic_preserve_table(img, original_text)
                if was_table:
                    logging.info("Table-like layout detected for %s", name)

                translated_text = None
                if translate and original_text and original_text.strip():
                    translated_text = translate_text(original_text, target_lang=dest_lang)

                return name, {
                    "original_text": formatted_text if formatted_text is not None else None,
                    "translated_text": translated_text,
                }
        except Exception as e:
            logging.warning("Skipping %s: %s", name, e)
            return None

    total = len(image_files)
    start_time = time.time()

    # progress bar setup
    use_tqdm = show_progress and (tqdm is not None)
    if show_progress and not use_tqdm:
        logging.info("tqdm not installed, using built-in progress reporting (no dependency required). To disable progress, pass show_progress=False or --no-progress.")

    # run in parallel (threads) by default
    if parallel and total > 1:
        max_workers = workers or min(32, (os.cpu_count() or 1) + 4)
        completed = 0
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for path in image_files:
                futures.append(ex.submit(_process_one, path))

            if use_tqdm:
                for fut in tqdm(as_completed(futures), total=total, desc="OCR", unit="file"):
                    res = fut.result()
                    completed += 1
                    if res:
                        name, item = res
                        results[name] = item
            else:
                # Interpolated progress: poll for completed futures with a short timeout
                remaining = set(futures)
                last_completion_time = start_time
                # Loop until all futures complete
                while remaining:
                    done, not_done = wait(remaining, timeout=0.25, return_when=FIRST_COMPLETED)
                    # process any newly finished futures
                    for fut in list(done):
                        try:
                            res = fut.result()
                        except Exception:
                            res = None
                        if res:
                            name, item = res
                            results[name] = item
                        completed += 1
                        remaining.remove(fut)
                        last_completion_time = time.time()

                    # compute fractional progress estimate
                    elapsed = time.time() - start_time
                    avg = elapsed / completed if completed > 0 else None
                    if avg:
                        # estimate how many files worth of progress have happened since last completion
                        time_since_last = time.time() - last_completion_time
                        fractional_add = min(len(remaining), time_since_last / avg)
                        fractional_completed = completed + fractional_add
                    else:
                        fractional_completed = 0

                    pct = int((fractional_completed / total) * 100) if total else 100
                    remaining_secs = (total - fractional_completed) * (avg if avg else 0)
                    eta = time.strftime('%H:%M:%S', time.gmtime(max(0, remaining_secs)))
                    print(f"Progress: {fractional_completed:.2f}/{total} ({pct}%) ETA: {eta}", end='\r')

                # final newline
                print()
    else:
        # sequential path (also used when parallel disabled)
        completed = 0
        for path in image_files:
            res = _process_one(path)
            completed += 1
            if show_progress and not use_tqdm:
                elapsed = time.time() - start_time
                avg = elapsed / completed
                remaining = avg * (total - completed)
                pct = int((completed / total) * 100)
                eta = time.strftime('%H:%M:%S', time.gmtime(remaining))
                print(f"Progress: {completed}/{total} ({pct}%) ETA: {eta}", end='\r')
            if res:
                name, item = res
                results[name] = item
        if show_progress and not use_tqdm:
            print()

    # write outputs if requested
    if output_json:
        try:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info("Wrote JSON results to %s", output_json)
        except Exception as e:
            logging.error("Failed to write JSON: %s", e)

    if output_csv:
        try:
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "original_text", "translated_text"])
                for name, item in results.items():
                    writer.writerow([name, item.get("original_text"), item.get("translated_text")])
            logging.info("Wrote CSV results to %s", output_csv)
        except Exception as e:
            logging.error("Failed to write CSV: %s", e)

    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk OCR processor (eng+deu default)")
    # default to a common local folder to make the CLI ergonomic
    p.add_argument("--input_dir", default="./images", help="Path to folder with images (default: ./images)")
    p.add_argument("--output_json", default="ocr_results.json", help="JSON output path")
    p.add_argument("--output_csv", default=None, help="CSV/TSV output path (optional)")
    p.add_argument("--languages", default="eng+deu", help="Tesseract languages, e.g. 'eng+deu'")
    p.add_argument("--translate", action="store_true", help="Enable automatic translation")
    p.add_argument("--dest_lang", default="en", help="Destination language for translation (e.g. 'en')")
    p.add_argument("--no-parallel", action="store_true", help="Disable parallel processing (default: parallel enabled)")
    p.add_argument("--workers", type=int, default=None, help="Number of worker threads to use when parallel (default: auto)")
    p.add_argument("--no-progress", action="store_true", help="Disable progress reporting to avoid dependency overhead (default: progress enabled)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # Friendly warnings for common defaults
    # Warn if the input folder doesn't exist or is empty so users know to populate it
    if not os.path.exists(args.input_dir) or not os.path.isdir(args.input_dir):
        logging.warning(
            "Input folder '%s' does not exist. Populate '%s' with images or specify --input_dir to a different folder.",
            args.input_dir,
            args.input_dir,
        )
    else:
        # check for supported image files
        images = find_image_files(args.input_dir)
        if not images:
            logging.warning(
                "Input folder '%s' contains no supported image files. Populate it or specify another folder via --input_dir.",
                args.input_dir,
            )

    # JSON is the default output; remind users how to enable CSV output if desired
    if not args.output_csv:
        logging.warning(
            "Using JSON output by default (path: %s). Use --output_csv to also export CSV/TSV if you prefer tabular output.",
            args.output_json,
        )
    results = process_folder(
        args.input_dir,
        output_json=args.output_json,
        output_csv=args.output_csv,
        languages=args.languages,
        translate=args.translate,
        dest_lang=args.dest_lang,
        parallel=not args.no_parallel,
        workers=args.workers,
        show_progress=not args.no_progress,
    )

    # For small jobs, also print a summary to the console
    if len(results) <= 10:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
