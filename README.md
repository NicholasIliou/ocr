# Multi-Language OCR Processor

Small Python tool to perform batch OCR over a folder of images, supporting English and German by default. Optional translation can be enabled.

Features
- Processes common image formats (.png, .jpg, .jpeg, .tiff, .bmp, .webp)
- Uses Tesseract via `pytesseract` to extract printed text (default `eng+deu`)
- Heuristic attempt to preserve table-like layout (inserts tabs where column gaps are detected)
- Optional translation using `deep-translator` (GoogleTranslator)
- Outputs results as `ocr_results.json` and optionally CSV

Installation

1. Install Tesseract OCR engine on your system:
   - Windows: download from https://github.com/tesseract-ocr/tesseract and add to PATH
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr` (and tesseract language packs as needed)

2. Create a virtualenv and install Python dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

Usage

```powershell
python ocr_processor.py --input_dir ./images --output_json ocr_results.json --output_csv ocr_results.csv

# With translation to English
python ocr_processor.py --input_dir ./images --translate --dest_lang en
```

Notes
- The script requires the Tesseract binary to be installed and available in PATH.
- Translation is optional and uses networked Google translation via `deep-translator`.
- Table preservation is heuristic — it works for many tabular screenshots but is not perfect.

Example JSON output structure

```json
{
  "invoice_01.png": {
    "original_text": "Rechnung Nr. 25 ...",
    "translated_text": "Invoice No. 25 ..."
  }
}
```

Next steps / stretch goals
- Add EasyOCR fallback for handwriting
- Use OpenCV-based table detection for better table extraction
- Add parallel processing and a progress bar
