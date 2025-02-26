# Award Document Search

A Python-based tool for extracting, searching, and classifying text content from PDF, MS Word, and Excel documents with support for both standard text extraction and OCR capabilities.

## Features

- PDF text extraction using PyMuPDF for machine-readable documents
- OCR support using Tesseract for scanned documents
- Keyword-based search functionality
- Support for exporting extracted text to files

## Requirements

- Python >= 3.11
- Tesseract OCR (for OCR functionality)

## Installation

This project uses `uv` for dependency management. To install `uv`:

```bash
pip install uv
```

To install the project's dependencies:

```bash
uv sync
```

## Project Structure

- `search.py`: Main script for PDF text extraction and search
- `extract.py`: Additional extraction utilities
- `readdocx.py`: Support for MS Word file processing
- `spellcheck.py`: Spell checking functionality
- `tokenize_.py`: Text tokenization utilities

## Usage

The main functionality is provided through the `search.py` script. Example usage:

```bash
uv run python search.py --pdf-path path/to/document.pdf --keywords path/to/keywords.txt
```
