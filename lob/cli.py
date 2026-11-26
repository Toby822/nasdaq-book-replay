"""Inspect saved order book replays without loading the original CSVs."""
import argparse
import json
from pathlib import Path

from .mapped import open_mapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    seek = commands.add_parser("seek")
    seek.add_argument("timestamp_ps", type=int, help="Integer picoseconds since midnight")
    seek.add_argument("--before", action="store_true")
    replay_cmd = commands.add_parser("replay")
    replay_cmd.add_argument("start", type=int, help="Inclusive row index")
    replay_cmd.add_argument("stop", type=int, help="Exclusive row index")
    args = parser.parse_args()
    book = open_mapped(args.archive)
    try:
        if args.command == "seek":
            row = book.as_of(args.timestamp_ps, before=args.before)
            print(json.dumps(None if row is None else row.tolist()))
        else:
            for index, row in enumerate(book.iter_rows(args.start, args.stop), args.start):
                print(json.dumps({"index": index, "time_ps": int(book.times[index]), "depth": row.tolist()}))
    except IndexError as error:
        parser.error(f"Row range outside the archive: {error}")


if __name__ == "__main__":
    main()
