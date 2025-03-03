# Award Document Search

A Python-based tool for searching and analyzing PDF and Microsoft Word documents using keyword-based search with support for both text extraction and OCR capabilities.

## Features

- PDF text extraction with OCR support using Tesseract
- Microsoft Word (.docx) document processing
- Keyword-based search
- CSV output format for structured results

## Requirements

- Python >= 3.11
- Tesseract OCR (for PDF OCR functionality)
- Antiword (for opening old MS-Word files)
- Enchant (for spell checking)

```bash
brew update
brew install caskroom/cask/brew-cask
brew cask install xquartz
brew install poppler antiword unrtf tesseract swig
brew install enchant

# Python doesn't always find the enchant library, so we need to set the PYENCHANT_LIBRARY_PATH environment variable
echo 'export PYENCHANT_LIBRARY_PATH=$(brew --prefix enchant)/lib/libenchant-2.dylib' >> ~/.zshrc
```

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
usage: award [-h] --input INPUT --keywords KEYWORDS --output OUTPUT

Search for keywords in PDF, DOC, and DOCX files. Output is a CSV file.

options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input directory
  --keywords KEYWORDS, -k KEYWORDS
                        Keywords file path
  --output OUTPUT, -o OUTPUT
                        Output csv file path
```

## Features

- `main.py`: entry point
- `utils.py`: Miscellaneous utilities for document and search processing
- `extract.py`: Zip file extraction utilities
- `spellcheck.py`: Spell checking functionality

## Usage

The tool can handle PDF or MS-Word documents seamlessly through the `award` command.

```bash
uv run award --input path/to/pdf-files --keywords path/to/keywords.txt --output results.csv
```

The tool outputs results in CSV format containing matched keywords and their context within the documents. For example:

<table border="1">
    <tr>
        <th>Code</th>
        <th>Description</th>
        <th>keyword1</th>
        <th>keyword2</th>
        <th>keyword3</th>
    </tr>
    <tr>
        <td>001</td>
        <td>File-name1</td>
        <td>FALSE</td>
        <td>FALSE</td>
        <td>FALSE</td>
    </tr>
    <tr>
        <td>002</td>
        <td>File-name2</td>
        <td>TRUE</td>
        <td>TRUE</td>
        <td>TRUE</td>
    </tr>
    <tr>
        <td>003</td>
        <td>File-name3</td>
        <td>FALSE</td>
        <td>FALSE</td>
        <td>FALSE</td>
    </tr>
</table>