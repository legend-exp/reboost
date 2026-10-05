from __future__ import annotations

import argparse
import logging
from typing import Literal

import dbetto

from ..log_utils import setup_log
from ..utils import _check_input_file, _check_output_file

log = logging.getLogger(__name__)


def optmap_cli() -> None:
    parser = argparse.ArgumentParser(
        prog="reboost-optmap",
        description="%(prog)s: create and manipulate optical maps for usage with reboost",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
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

    # COMMAND: build map file from stp tier
    create_parser = subparsers.add_parser("create", help="build optical map from stp file(s)")
    create_parser.add_argument(
        "--settings",
        action="store",
        help="""Select a config file for binning.""",
        required=True,
    )
    create_parser.add_argument(
        "--detectors",
        help=(
            "file that contains a list of detector ids that will be produced as additional output maps."
            + "By default, all channels will be included."
        ),
    )
    create_parser_det_group = create_parser.add_mutually_exclusive_group(required=True)
    create_parser_det_group.add_argument(
        "--geom",
        help="GDML geometry file",
    )
    create_parser.add_argument(
        "--n-procs",
        "-N",
        type=int,
        default=1,
        help="number of worker processes to use. default: %(default)e",
    )
    create_parser.add_argument(
        "--check",
        action="store_true",
        help="""Check map statistics after creation. default: %(default)s""",
    )
    create_parser.add_argument(
        "input", help="input stp or optmap-evt LH5 file", metavar="INPUT_EVT", nargs="+"
    )
    create_parser.add_argument("output", help="output map LH5 file", metavar="OUTPUT_MAP")

    # COMMAND: view maps
    view_parser = subparsers.add_parser(
        "view",
        help="view optical map (arrows: navigate slices/axes, 'c': channel selector)",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Interactively view optical maps stored in LH5 files.\n\n"
            "Keyboard controls:\n"
            "  left/right  - previous/next slice along the current axis\n"
            "  up/down     - switch slicing axis (x, y, z)\n"
            "  c           - open channel selector overlay to switch detector map\n\n"
            "Display notes:\n"
            "  - Cells where no primary photons were simulated are shown in white.\n"
            "  - Cells where no photons were detected are shown in grey.\n"
            "  - Cells with values above the colormap maximum are shown in red.\n"
            "  - Use --hist to choose which histogram to display. 'prob_unc_rel' shows the\n"
            "    relative uncertainty prob_unc / prob where defined.\n"
            "  - Use --divide to show the ratio of two map files (this/other)."
        ),
        epilog=(
            "Examples:\n"
            "  reboost-optmap view mymap.lh5\n"
            "  reboost-optmap view mymap.lh5 --channel _1067205\n"
            "  reboost-optmap view mymap.lh5 --hist prob_unc_rel --min 0 --max 1\n"
            "  reboost-optmap view mymap.lh5 --divide other.lh5 --title 'Comparison'"
        ),
    )
    view_parser.add_argument("input", help="input map LH5 file", metavar="INPUT_MAP")
    view_parser.add_argument(
        "--channel",
        action="store",
        default="all",
        help="channel to display ('all' or '_<detid>'). Press 'c' in the viewer to switch. default: %(default)s",
    )
    view_parser.add_argument(
        "--hist",
        choices=("_nr_gen", "_nr_det", "prob", "prob_unc", "prob_unc_rel"),
        action="store",
        default="prob",
        help="select optical map histogram to show. default: %(default)s",
    )
    view_parser.add_argument(
        "--divide",
        action="store",
        help="divide by another map file before display (ratio). default: none",
    )
    view_parser.add_argument(
        "--min",
        default=1e-4,
        type=(lambda s: s if s == "auto" else float(s)),
        help="colormap min value; use 'auto' for automatic scaling. default: %(default)e",
    )
    view_parser.add_argument(
        "--max",
        default=1e-2,
        type=(lambda s: s if s == "auto" else float(s)),
        help="colormap max value; use 'auto' for automatic scaling. default: %(default)e",
    )
    view_parser.add_argument("--title", help="title of figure. default: stem of filename")

    # COMMAND: merge maps
    merge_parser = subparsers.add_parser("merge", help="merge optical maps")
    merge_parser.add_argument("input", help="input map LH5 files", metavar="INPUT_MAP", nargs="+")
    merge_parser.add_argument("output", help="output map LH5 file", metavar="OUTPUT_MAP")
    merge_parser.add_argument(
        "--settings",
        action="store",
        help="""Select a config file for binning.""",
        required=True,
    )
    merge_parser.add_argument(
        "--n-procs",
        "-N",
        type=int,
        default=1,
        help="number of worker processes to use. default: %(default)e",
    )
    merge_parser.add_argument(
        "--check",
        action="store_true",
        help="""Check map statistics after creation. default: %(default)s""",
    )

    # COMMAND: check map
    check_parser = subparsers.add_parser("check", help="check optical maps")
    check_parser.add_argument("input", help="input map LH5 file", metavar="INPUT_MAP")

    # COMMAND: patch a region of a map with a separately simulated one
    patch_parser = subparsers.add_parser(
        "patch", help="replace a region of an optical map with a patch map"
    )
    patch_parser.add_argument("input", help="input map LH5 file", metavar="INPUT_MAP")
    patch_parser.add_argument(
        "patch", help="patch map LH5 file, replacing the region it covers", metavar="PATCH_MAP"
    )
    patch_parser.add_argument("output", help="output map LH5 file", metavar="OUTPUT_MAP")

    # COMMAND: rebin maps
    rebin_parser = subparsers.add_parser("rebin", help="rebin optical maps")
    rebin_parser.add_argument("input", help="input map LH5 files", metavar="INPUT_MAP")
    rebin_parser.add_argument("output", help="output map LH5 file", metavar="OUTPUT_MAP")
    rebin_parser.add_argument("--factor", type=int, help="integer scale-down factor")

    args = parser.parse_args()

    log_level = (None, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    setup_log(log_level)

    # the subcommand modules are imported here, so that a subcommand does not pay
    # the import cost of the others

    # COMMAND: build map file from evt tier
    if args.command == "create":
        from .create import create_optical_maps  # noqa: PLC0415

        _check_input_file(parser, args.input)
        _check_output_file(parser, args.output)

        # load settings for binning from config file.
        _check_input_file(parser, args.input, "settings")
        settings = dbetto.utils.load_dict(args.settings)

        chfilter: tuple[str | int, ...] | Literal["*"] = "*"
        if args.detectors is not None:
            # load detector ids from a JSON/YAML array
            chfilter = tuple(dbetto.utils.load_dict(args.detectors))

        create_optical_maps(
            args.input,
            settings,
            args.bufsize,
            chfilter=chfilter,
            output_lh5_fn=args.output,
            check_after_create=args.check,
            n_procs=args.n_procs,
            geom_fn=args.geom,
        )

    # COMMAND: view maps
    if args.command == "view":
        from .mapview import view_optmap  # noqa: PLC0415

        _check_input_file(parser, args.input)
        if args.divide is not None:
            _check_input_file(parser, args.divide)
        view_optmap(
            args.input,
            args.channel,
            args.divide,
            cmap_min=args.min,
            cmap_max=args.max,
            title=args.title,
            histogram_choice=args.hist,
        )

    # COMMAND: merge maps
    if args.command == "merge":
        from .create import merge_optical_maps  # noqa: PLC0415

        # load settings for binning from config file.
        _check_input_file(parser, args.input, "settings")
        settings = dbetto.utils.load_dict(args.settings)

        _check_input_file(parser, args.input)
        _check_output_file(parser, args.output)
        merge_optical_maps(
            args.input, args.output, settings, check_after_create=args.check, n_procs=args.n_procs
        )

    # COMMAND: check maps
    if args.command == "check":
        from .create import check_optical_map  # noqa: PLC0415

        _check_input_file(parser, args.input)
        check_optical_map(args.input)

    # STEP 1e: patch map
    if args.command == "patch":
        from .create import patch_optical_map_lh5  # noqa: PLC0415

        _check_input_file(parser, [args.input, args.patch])
        _check_output_file(parser, args.output)
        patch_optical_map_lh5(args.input, args.patch, args.output)

    # COMMAND: rebin maps
    if args.command == "rebin":
        from .create import rebin_optical_maps  # noqa: PLC0415

        _check_input_file(parser, args.input)
        _check_output_file(parser, args.output)
        rebin_optical_maps(args.input, args.output, args.factor)
