"""Deployment script"""

import sys
import shutil
from pathlib import Path


PROJECT_PATH = Path(__file__).parent / 'htmplate'


def main():
    deployment_path = Path(sys.argv[1]) / 'htmplate'
    tmp = shutil.ignore_patterns("__pycache__")
    shutil.copytree(PROJECT_PATH, deployment_path, ignore=tmp)


if __name__ == '__main__':
    main()
