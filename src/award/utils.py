
import io
import re
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image
from tqdm import tqdm


def get_document_code_description(file_path: Path) -> str:
    index = re.search(r"[^\d_]", file_path).start()
    if index > 0:
        return file_path[: index - 1], file_path[index:]
    return None


def parse_code_and_description(df: pd.DataFrame) -> pd.DataFrame:
    leftmost_columns = ["Code", "Description"]
    df[leftmost_columns] = df["File"].apply(
        lambda x: pd.Series(get_document_code_description(Path(x).stem))
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
    # breakpoint()    # if text is too short, try OCR
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


def search_files(input_dir, keywords_file, output_file) -> None:
    keywords = get_keywords(keywords_file)
    results = {}

    input_path = Path(input_dir)
    for file in tqdm(list(input_path.glob("**/*.pdf"))):
        if not file.is_file():
            continue

        text = extract_text_from_pdf(file, totxt=False)
        results[str(file)] = search_keywords_in_text(text, keywords)
    write_results_to_csv(results, output_file)


