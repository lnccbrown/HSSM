# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

import argparse
# from pprint import pp
from pathlib import Path
import json

import textract
# from nltk.stem import WordNetLemmatizer

from tokenize_ import get_tokens
from searchpdf import get_keywords


def read_doc(file_path):
    text = textract.process(file_path).decode("utf-8")
    return text

def filter_tokens_in_keywords(tokens, keywords):
    return [t for t in tokens if t[0].lower() in keywords]

def search_files(input_dir, keywords_file, output_file):
    keywords = get_keywords(keywords_file)
    results = {}
    
    input_path = Path(input_dir)
    for file in input_path.glob("**/*.doc*"):
        if not file.is_file():
            continue
        
        text = read_doc(file)
        tokens = get_tokens(text)
        found = filter_tokens_in_keywords(tokens, keywords)
        
        if found:
            results[str(file)] = found
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

def add_docx_subparser(subparsers):
    """Add DOCX search arguments to an argparse subparsers object."""
    description = "Check inclusion of keywords in MS-word files."
    docx_parser = subparsers.add_parser('docx', help='Search in DOC and DOCX files', description=description)
    docx_parser.add_argument('--input', '-i', required=True, help='Input directory')
    docx_parser.add_argument('--keywords', '-k', required=True, help='Keywords file path')
    docx_parser.add_argument('--output', '-o', required=True, help='Output json file path')
    return docx_parser

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    docx_parser = add_docx_subparser(subparsers)
    
    args = parser.parse_args()
    if args.command == 'docx':
        search_files(args.input, args.keywords, args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
