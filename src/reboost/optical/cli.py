from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from pathlib import Path

import colorlog

log = logging.getLogger(__name__)


def optical_cli() -> None:
    parser = argparse.ArgumentParser(
        prog="reboost-optical",
        description="%(prog)s command line interface",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="""Increase the program verbosity""",
    )

    parser.add_argument(
        "--bufsize",
        action="store",
        type=int,
        default=int(5e6),
        help="""Row count for input table buffering (only used if applicable). default: %(default)e""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # STEP 1: build evt file from hit tier
    evt_parser = subparsers.add_parser("evt", help="build evt file from remage hit file")
    evt_parser.add_argument(
        "--detectors",
        help="file that contains a list of detector ids that are part of the input file",
        required=True,
    )
    evt_parser.add_argument("input", help="input hit LH5 file", metavar="INPUT_HIT")
    evt_parser.add_argument("output", help="output evt LH5 file", metavar="OUTPUT_EVT")

    # STEP 2a: build map file from evt tier
    map_parser = subparsers.add_parser("createmap", help="build optical map from evt file(s)")
    map_parser.add_argument(
        "--settings",
        action="store",
        help="""Select a config file for binning.""",
        required=True,
    )
    map_parser.add_argument("input", help="input evt LH5 file", metavar="INPUT_EVT", nargs="+")
    map_parser.add_argument("output", help="output map LH5 file", metavar="OUTPUT_MAP")

    # STEP 2b: view maps
    mapview_parser = subparsers.add_parser("viewmap", help="view optical map")
    mapview_parser.add_argument("input", help="input map LH5 file", metavar="INPUT_MAP")
    mapview_parser.add_argument(
        "--channel",
        action="store",
        default="all",
        help="default: %(default)s",
    )
    mapview_parser.add_argument(
        "--min",
        default=1e-3,
        type=float,
        help="colormap min value. default: %(default)e",
    )
    mapview_parser.add_argument(
        "--max",
        default=1e-2,
        type=float,
        help="colormap max value. default: %(default)e",
    )
    mapview_parser.add_argument("--title", help="title of figure. default: stem of filename")

    # STEP 2c: merge maps
    mapmerge_parser = subparsers.add_parser("mergemap", help="merge optical maps")
    mapmerge_parser.add_argument(
        "input", help="input map LH5 files", metavar="INPUT_MAP", nargs="+"
    )
    mapmerge_parser.add_argument("output", help="output map LH5 file", metavar="OUTPUT_MAP")
    mapmerge_parser.add_argument(
        "--settings",
        action="store",
        help="""Select a config file for binning.""",
        required=True,
    )

    # STEP 3: convolve with hits from non-optical simulations
    convolve_parser = subparsers.add_parser(
        "convolve", help="convolve non-optical hits with optical map"
    )
    convolve_parser.add_argument(
        "--material",
        action="store",
        choices=("lar", "pen", "fib"),
        default="lar",
        help="default: %(default)s",
    )
    convolve_parser.add_argument(
        "--map",
        action="store",
        required=True,
        metavar="INPUT_MAP",
        help="input map LH5 file",
    )
    convolve_parser.add_argument(
        "--edep",
        action="store",
        required=True,
        metavar="INPUT_EDEP",
        help="input non-optical LH5 hit file",
    )
    convolve_parser.add_argument(
        "--edep-lgdo",
        action="store",
        required=True,
        metavar="LGDO_PATH",
        help="path to LGDO inside non-optical LH5 hit file (e.g. /hit/detXX)",
    )
    convolve_parser.add_argument("--output", help="output hit LH5 file", metavar="OUTPUT_HIT")

    args = parser.parse_args()

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(name)s [%(levelname)s] %(message)s")
    )
    logger = logging.getLogger("reboost.optical")
    logger.addHandler(handler)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # STEP 1: build evt file from hit tier
    if args.command == "evt":
        from reboost.optical.evt import build_optmap_evt

        _check_input_file(parser, args.detectors)
        _check_input_file(parser, args.input)
        _check_output_file(parser, args.output)

        # load detector ids from a JSON array
        with Path.open(Path(args.detectors)) as detectors_f:
            detectors = json.load(detectors_f)

        build_optmap_evt(args.input, args.output, detectors, args.bufsize)

    # STEP 2a: build map file from evt tier
    if args.command == "createmap":
        from reboost.optical.create import create_optical_maps
        from reboost.optical.evt import read_optmap_evt

        _check_input_file(parser, args.input)
        _check_output_file(parser, args.output)

        # load settings for binning from config file.
        _check_input_file(parser, args.input, "settings")
        with Path.open(Path(args.settings)) as settings_f:
            settings = json.load(settings_f)

        optmap_events = read_optmap_evt(args.input, args.bufsize)
        create_optical_maps(
            optmap_events,
            settings,
            chfilter=(),
            output_lh5_fn=args.output,
        )

    # STEP 2b: view maps
    if args.command == "viewmap":
        from reboost.optical.mapview import view_optmap

        _check_input_file(parser, args.input)
        view_optmap(
            args.input, args.channel, cmap_min=args.min, cmap_max=args.max, title=args.title
        )

    # STEP 2c: merge maps
    if args.command == "mergemap":
        from reboost.optical.create import merge_optical_maps

        # load settings for binning from config file.
        _check_input_file(parser, args.input, "settings")
        with Path.open(Path(args.settings)) as settings_f:
            settings = json.load(settings_f)

        _check_input_file(parser, args.input)
        _check_output_file(parser, args.output)
        merge_optical_maps(args.input, args.output, settings)

    # STEP 3: convolve with hits from non-optical simulations
    if args.command == "convolve":
        from reboost.optical.convolve import convolve

        _check_input_file(parser, [args.map, args.edep])
        _check_output_file(parser, args.output)
        convolve(args.map, args.edep, args.edep_lgdo, args.material, args.output, args.bufsize)


def _check_input_file(parser, file: str | Iterable[str], descr: str = "input"):
    file = (file,) if isinstance(file, str) else file
    not_existing = [f for f in file if not Path(f).exists()]
    if not_existing != []:
        parser.error(f"{descr} file(s) {''.join(not_existing)} missing")


def _check_output_file(parser, file: str):
    if Path(file).exists():
        parser.error(f"output file {file} already exists")
