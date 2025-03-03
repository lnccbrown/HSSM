import io
import logging
import re
from pathlib import Path
from pprint import pformat

import pymupdf
import pandas as pd
import pytesseract
import textract
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_document_code_description(file_path: Path | str)  -> tuple:
    file_path = Path(file_path)
    stem = file_path.stem
    match = re.search(r"[^\d_]", stem)
    if match:
        index = match.start() - 1 if match.start() > 1 else None
        code = stem[:index] if index else None
        description = stem[index + 1 :] if index is not None else stem
    else:
        code = stem
        description = None
    return code, description


def parse_code_and_description(df: pd.DataFrame) -> pd.DataFrame:
    leftmost_columns = ["Code", "Description"]
    df[leftmost_columns] = df["File"].apply(
        lambda x: pd.Series(get_document_code_description(x))
    )
    df = df.drop(columns=["File"])
    df = df[
        leftmost_columns + [col for col in df.columns if col not in leftmost_columns]
    ]
    return df


def write_results_to_csv(results: dict, output_file: str) -> None:
    df = pd.DataFrame.from_dict(results, orient="index").reset_index()
    df = df.rename(columns={"index": "File"})
    df = parse_code_and_description(df)
    df = df.sort_values(by=["Code", "Description"])
    df.to_csv(output_file, index=False)


def get_keywords(file_path: Path) -> tuple:
    with open(file_path, "r") as file:
        lines = tuple(line.strip().lower() for line in file.readlines())
    return lines


def export_text_to_file(text: str, output_path: Path) -> None:
    with open(output_path, "w") as file:
        file.write(text)


def search_keywords_in_text(text: str, keywords: tuple) -> dict:
    results = {
        keyword: bool(re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE))
        for keyword in keywords
    }
    return results


def extract_text_from_doc_simple(pdf_path: Path) -> tuple[str, int]:
    """Extract text content from a machine-readable PDF file.

    Args:
        pdf_path (Path): Path to the PDF file to process

    Returns:
        tuple: A tuple containing:
            - str: The extracted text from all pages concatenated
            - int: Total number of pages in the PDF
    """
    pdf_path = Path(pdf_path)
    document = pymupdf.open(pdf_path)
    page_count = document.page_count

    if page_count < 1:
        raise ValueError("No pages found. File may be corrupted or empty.")

    text = ""
    for page_num in range(page_count):
        page = document.load_page(page_num)
        page_text = page.get_text()
        text += page_text

    return text, document.page_count


def extract_text_from_pdf_ocr(pdf_path: Path) -> str:
    pdf_path = Path(pdf_path)
    text = ""
    document = pymupdf.open(pdf_path)
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        page_text = pytesseract.image_to_string(img)

        text += page_text

    return text


def extract_text_from_doc(file_path: Path, totxt=False) -> str:
    extension = file_path.suffix.lower()
    # first try simple if document is machine readable
    if extension != ".doc":
        text, page_count = extract_text_from_doc_simple(file_path)
        logging.info("Extracted %d pages from %s", page_count, file_path)

        # if text is too short, try OCR
        if len(text.split("\n")) < 10 * page_count:
            logging.info("Warning: File is not machine readable. Trying OCR.")
            text = extract_text_from_pdf_ocr(file_path)
            # TODO: apply spellcheck to OCR text

    else:
        text = read_doc(file_path)
        logging.info("Succesfully processed %s", file_path)

    if totxt:
        out_name = file_path.stem
        out_name = Path(out_name).with_suffix(".txt")
        export_text_to_file(text, out_name)
        logging.info("Text extracted and saved to %s", out_name)

    return text


def read_doc(file_path):
    text = textract.process(file_path).decode("utf-8")
    return text


def find_files_with_extensions(input_dir: Path, extensions: list) -> list:
    patterns = [f"**/*{ext}" for ext in extensions]
    return sorted([file for pattern in patterns for file in input_dir.glob(pattern)])


def search_files(
    input_dir, keywords_file, output_file, extensions=(".pdf", ".docx", ".doc")
) -> None:
    keywords = get_keywords(keywords_file)
    results = {}
    failures = []
    input_path = Path(input_dir)
    all_paths = find_files_with_extensions(input_path, extensions)
    for file in tqdm(all_paths):
        if not file.is_file():
            continue
        try:
            text = extract_text_from_doc(file, totxt=False)
            results[str(file)] = search_keywords_in_text(text, keywords)
        except Exception as e:
            logging.error("Error processing %s: %s", file, e)
            failures.append(file)
            continue

    if failures:
        logging.warning(
            "Failed to process %d files:\n%s",
            len(failures),
            pformat([str(f) for f in failures]),
        )

    if results:
        write_results_to_csv(results, output_file)
