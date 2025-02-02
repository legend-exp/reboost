from __future__ import annotations

import argparse
import logging

import dbetto

from reboost.build_glm import build_glm
from reboost.build_hit import build_hit
from reboost.utils import _check_input_file, _check_output_file

from .log_utils import setup_log

log = logging.getLogger(__name__)


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="reboost",
        description="%(prog)s command line interface",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="""Increase the program verbosity""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # glm parser
    glm_parser = subparsers.add_parser("build-glm", help="build glm file from remage stp file")

    glm_parser.add_argument(
        "--stp_file", "-s", required=True, type=str, help="Path to the stp file."
    )
    glm_parser.add_argument(
        "--glm_file", "-g", required=True, type=str, help="Path to the glm file "
    )

    # optional args
    glm_parser.add_argument(
        "--out_table_name", "-n", type=str, default="glm", help="Output table name."
    )
    glm_parser.add_argument("--id_name", "-i", type=str, default="g4_evtid", help="ID column name.")
    glm_parser.add_argument(
        "--evtid_buffer", "-e", type=int, default=int(1e7), help="event id buffer size."
    )
    glm_parser.add_argument(
        "--stp_buffer", "-b", type=int, default=int(1e7), help="stp buffer size."
    )
    glm_parser.add_argument(
        "--overwrite", "-w", action="store_true", help="Overwrite the input file if it exists."
    )

    # hit parser
    hit_parser = subparsers.add_parser("build-hit", help="build hit file from remage stp file")

    hit_parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file ."
    )
    hit_parser.add_argument("--args", type=str, required=True, help="Path to args file.")
    hit_parser.add_argument(
        "--stp_files",
        nargs="+",
        type=str,
        required=True,
        help="List of stp files or a single stp file.",
    )
    hit_parser.add_argument(
        "--glm_files",
        nargs="+",
        type=str,
        required=True,
        help="List of glm files or a single glm file.",
    )
    hit_parser.add_argument(
        "--hit_files",
        nargs="*",
        type=str,
        required=True,
        help="List of output hit files or a single hit file",
    )

    # optional args
    hit_parser.add_argument("--start-evtid", type=int, default=0, help="Start event id.")
    hit_parser.add_argument(
        "--n-evtid", type=int, default=None, help="Number of event id to process."
    )
    hit_parser.add_argument("--in-field", type=str, default="stp", help="Input field name.")
    hit_parser.add_argument("--out-field", type=str, default="hit", help="Output field name.")
    hit_parser.add_argument("--buffer", type=int, default=int(5e6), help="Buffer size.")

    hit_parser.add_argument(
        "--overwrite", "-w", action="store_true", help="Overwrite the input file if it exists."
    )

    args = parser.parse_args()

    log_level = (None, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    setup_log(log_level)

    if args.command == "build-glm":
        # catch some cases
        _check_input_file(parser, args.stp_file)

        if args.overwrite is False:
            _check_output_file(parser, args.glm_file)

        msg = "Running build_glm with arguments: \n"
        msg += f"    glm file:       {args.glm_file}\n"
        msg += f"    stp file:       {args.stp_file}\n"
        msg += f"    out_table_name: {args.out_table_name}\n"
        msg += f"    evtid_name:     {args.id_name}\n"
        msg += f"    evtid_buffer:   {args.evtid_buffer}\n"
        msg += f"    stp_buffer:     {args.stp_buffer}"

        log.info(msg)

        build_glm(
            args.stp_file,
            args.glm_file,
            out_table_name=args.out_table_name,
            id_name=args.id_name,
            evtid_buffer=args.evtid_buffer,
            stp_buffer=args.stp_buffer,
        )

    elif args.command == "build-hit":
        _check_input_file(parser, args.stp_files)
        _check_input_file(parser, args.glm_files)

        if args.overwrite is False:
            _check_output_file(parser, args.hit_files)

        msg = "Running build_hit with arguments: \n"
        msg += f"    config:         {args.config}\n"
        msg += f"    args:           {args.args}\n"
        msg += f"    glm files:      {args.glm_files}\n"
        msg += f"    stp files:      {args.stp_files}\n"
        msg += f"    hit files:      {args.hit_files}\n"
        msg += f"    start_evtid:    {args.start_evtid}\n"
        msg += f"    n_evtid:        {args.n_evtid}\n"
        msg += f"    in_field:       {args.in_field}\n"
        msg += f"    out_field:      {args.out_field}\n"
        msg += f"    buffer:         {args.buffer}"
        log.info(msg)

        build_hit(
            config=args.config,
            args=dbetto.AttrsDict(dbetto.utils.load_dict(args.args)),
            stp_files=args.stp_files,
            glm_files=args.glm_files,
            hit_files=args.hit_files,
            start_evtid=args.start_evtid,
            n_evtid=args.n_evtid,
            in_field=args.in_field,
            out_field=args.out_field,
            buffer=args.buffer,
        )
