# Award Document Search

A Python-based tool for searching and analyzing PDF and Microsoft Word documents using keyword-based search with support for both text extraction and OCR capabilities.

## Features

- PDF text extraction with OCR support using Tesseract
- Microsoft Word (.docx) document processing
- Keyword-based search with configurable output
- JSON output format for structured results
- Spell checking and text tokenization utilities
- Command-line interface with subcommands for different document types

## Requirements

- Python >= 3.11
- Tesseract OCR (for PDF OCR functionality)

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

- `main.py`: Entry point with command-line interface
- `searchpdf.py`: PDF document search functionality
- `searchdocx.py`: Microsoft Word document search functionality
- `extract.py`: Text extraction utilities
- `spellcheck.py`: Spell checking functionality
- `tokenize_.py`: Text tokenization utilities

## Usage

The tool provides a command-line interface with subcommands for different document types:

For PDF documents:
```bash
uv run python main.py pdf --input path/to/pdf --keywords path/to/keywords.txt --output results.json
```

For DOCX documents:
```bash
uv run python main.py docx --input path/to/document.docx --keywords path/to/keywords.txt --output results.json
```

The tool outputs results in JSON format containing matched keywords and their context within the documents.
