import argparse


from .utils import search_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for keywords in PDF, DOC, and DOCX files. Output is a CSV file."
    )
    parser.add_argument("--input", "-i", required=True, help="Input directory")
    parser.add_argument("--keywords", "-k", required=True, help="Keywords file path")
    parser.add_argument("--output", "-o", required=True, help="Output csv file path")

    args = parser.parse_args()
    search_files(args.input, args.keywords, args.output)


if __name__ == "__main__":
    main()
