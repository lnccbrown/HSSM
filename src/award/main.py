import argparse
from .searchdocx import search_files as search_docx, add_docx_subparser
from .searchpdf import search_files as search_pdf, add_pdf_subparser

def main():
    parser = argparse.ArgumentParser(description='Search for keywords in PDF and DOCX files')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Add DOCX search subcommand using the parser from searchdocx.py
    add_docx_subparser(subparsers)

    # Add PDF search subcommand using the parser from searchpdf.py
    add_pdf_subparser(subparsers)

    args = parser.parse_args()

    if args.command == 'docx':
        search_docx(args.input, args.keywords, args.output)
    elif args.command == 'pdf':
        search_pdf(args.input, args.keywords, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
