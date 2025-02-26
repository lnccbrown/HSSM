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

## Entry point with command-line interface

```bash
$ uv run award --help
usage: award [-h] {docx,pdf} ...

Search for keywords in PDF and DOCX files

positional arguments:
  {docx,pdf}  Commands
    docx      Search in DOC and DOCX files
    pdf       Search in PDF files

options:
  -h, --help  show this help message and exit
```

As shown in the example, the tool provides subcommands for different document types (`docx` and `pdf`). For the `pdf` subcommand,
```bash
$ uv run award pdf --help
usage: award pdf [-h] --input INPUT --keywords KEYWORDS --output OUTPUT

Search for keywords in PDF files. Output is a json file where the keys are the file names and the values are the results of the search.

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input directory
  --keywords KEYWORDS, -k KEYWORDS
                        Keywords file path
  --output OUTPUT, -o OUTPUT
                        Output json file path
```

## Features

- `main.py`: entry point
- `searchpdf.py`: PDF document search functionality
- `searchdocx.py`: Microsoft Word document search functionality
- `extract.py`: Text extraction utilities
- `spellcheck.py`: Spell checking functionality
- `tokenize_.py`: Text tokenization utilities

## Usage

The tool provides a command-line interface with subcommands for different document types:

For PDF documents:
```bash
uv run award pdf --input path/to/pdf --keywords path/to/keywords.txt --output results.json
```

For DOCX documents:
```bash
uv run award docx --input path/to/document.docx --keywords path/to/keywords.txt --output results.json
```

The tool outputs results in JSON format containing matched keywords and their context within the documents.
