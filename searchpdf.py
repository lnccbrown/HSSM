# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf"]
# ///


import fitz  # PyMuPDF
from pathlib import Path
import argparse
import re
from pprint import pp
import pytesseract
from PIL import Image
import io
import json
from tqdm import tqdm


def get_keywords(file_path: Path) -> tuple:
    with open(file_path, "r") as file:
        lines = tuple(line.strip().lower() for line in file.readlines())
    return lines


def export_text_to_file(text: str, output_path: Path) -> None:
    with open(output_path, "w") as file:
        file.write(text)


def extract_text_from_pdf_simple(pdf_path: Path) -> tuple[str, int]:
    """Extract text content from a machine-readable PDF file.

    Args:
        pdf_path (Path): Path to the PDF file to process

    Returns:
        tuple: A tuple containing:
            - str: The extracted text from all pages concatenated
            - int: Total number of pages in the PDF
    """
    pdf_path = Path(pdf_path)
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        page_text = page.get_text()
        text += page_text

    return text, document.page_count


def extract_text_from_pdf_ocr(pdf_path: Path) -> str:
    pdf_path = Path(pdf_path)
    text = ""
    document = fitz.open(pdf_path)
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        page_text = pytesseract.image_to_string(img)

        text += page_text

    return text


def extract_text_from_pdf(pdf_path: Path, totxt=True) -> str:
    # first try simple if pdf is machine readable
    text, page_count = extract_text_from_pdf_simple(pdf_path)
    print(f"Extracted {page_count} pages from {pdf_path}")
    # breakpoint()
    # if text is too short, try OCR
    if len(text.split("\n")) < 10 * page_count:
        print("Warning: File is not machine readable. Trying OCR.")
        text = extract_text_from_pdf_ocr(pdf_path)

        # TODO: apply spellcheck to OCR text

    if totxt:
        out_name = pdf_path.stem
        out_name = Path(out_name).with_suffix(".txt")
        export_text_to_file(text, out_name)
        print("Text extracted and saved to", out_name)

    return text


def search_keywords_in_text(text: str, keywords: tuple) -> dict:
    results = {
        keyword: bool(re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE))
        for keyword in keywords
    }
    return results


def main() -> None:
    description = "Search for keywords in PDF files. Output is a json file where the keys are the file names and the values are the results of the search."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "pdf_directory", type=str, help="Path to the directory containing PDF files."
    )
    parser.add_argument(
        "--keywords",
        type=Path,
        help="Path to the file containing keywords to search for.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output json file.",
        required=True,
    )

    args = parser.parse_args()

    pdf_directory = Path(args.pdf_directory)
    keywords = get_keywords(args.keywords)

    results = {
        pdf_file.name: search_keywords_in_text(
            extract_text_from_pdf(pdf_file), keywords
        )
        for pdf_file in pdf_directory.glob("*.pdf")
    }

    # serialize results to json file
    with open(args.output, "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    main()
