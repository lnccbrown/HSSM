# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf"]
# ///


import argparse

from utils import search_files


def add_pdf_subparser(subparsers):
    """Add PDF search arguments to an argparse subparsers object."""
    description = "Search for keywords in PDF files. Output is a json file where the keys are the file names and the values are the results of the search."
    pdf_parser = subparsers.add_parser(
        "pdf", help="Search in PDF files", description=description
    )
    pdf_parser.add_argument("--input", "-i", required=True, help="Input directory")
    pdf_parser.add_argument(
        "--keywords", "-k", required=True, help="Keywords file path"
    )
    pdf_parser.add_argument(
        "--output", "-o", required=True, help="Output json file path"
    )
    return pdf_parser


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    pdf_parser = add_pdf_subparser(subparsers)  # noqa: F841
    args = parser.parse_args()
    if args.command == "pdf":
        search_files(args.input, args.keywords, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
