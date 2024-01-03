import sys
import json
import argparse
from pathlib import Path

from .make import make


def error_exit(message: str):
    print("Error: " + message, file=sys.stderr)
    sys.exit(1)


def main():
    DEFAULT_OUT_PATH = './htmplate-out/'

    parser = argparse.ArgumentParser(description="Fill an html template with specified information.")
    parser.add_argument("template", type=str, help='HTML document template filename')
    parser.add_argument("info", type=str, help='JSON file with dictionaries')
    parser.add_argument("--dest", "-d", type=str, default=DEFAULT_OUT_PATH, help='A folder for output')
    parser.add_argument("--replace_old", "-r", default=False, action='store_true', help='Replace existing files')
    parser.add_argument("--filter", "-f", type=str, nargs='+', default=[],
                        help='Create templates only for specified keys')
    args = parser.parse_args(sys.argv[1:])

    template_path = Path(args.template).absolute()
    if template_path.suffix != '.htm':
        error_exit('Template file must be an html file with .htm extension')
    if template_path.is_dir() or not template_path.exists():
        error_exit('Template file does not exist or is a directory')
    with open(template_path) as f:
        template = f.read()

    values_path = Path(args.info).absolute()
    if values_path.suffix != '.json':
        error_exit('Values file must be a json file')
    if values_path.is_dir() or not values_path.exists():
        error_exit('Values file does not exist or is a directory')
    with open(values_path) as f:
        values = json.load(f)

    out_path = Path(args.dest).absolute()
    if args.dest == DEFAULT_OUT_PATH:
        if not out_path.is_dir():
            out_path.mkdir()
    if not out_path.is_dir():
        error_exit('Destination folder does not exist')

    for key, value in values.items():
        out_name = key + '.html'
        if args.filter != [] and key not in args.filter:
            continue
        if Path(out_path / out_name).exists() and not args.replace_old:
            continue
        generated_html = make(template, value)
        with open(out_path / out_name, 'w') as f:
            f.write(generated_html)


if __name__ == '__main__':
    main()
