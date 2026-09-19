from __future__ import annotations

import sys

from eeg_benchmark.tasks.cross_dataset import build_parser, run_parsed_args


def main() -> None:
    arguments = build_parser().parse_args(['cross-time', *sys.argv[1:]])
    run_parsed_args(arguments)


if __name__ == '__main__':
    main()
