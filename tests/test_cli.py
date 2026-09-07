from __future__ import annotations

from pathlib import Path

import lh5

from reboost.cli import cli


def test_cli(tmptestdir, remage_stp_file):
    test_file_dir = Path(__file__).parent / "hit"

    # test cli for build_glm
    cli(
        [
            "build-glm",
            "--id-name",
            "evtid",
            "-w",
            "--glm-file",
            f"{tmptestdir}/glm.lh5",
            "--stp-file",
            remage_stp_file,
        ]
    )

    glm = lh5.read("glm/det1", f"{tmptestdir}/glm.lh5").view_as("ak")
    assert glm.fields == ["evtid", "n_rows", "start_row"]

    # test cli for build_hit
    cli(
        [
            "build-hit",
            "--config",
            f"{test_file_dir}/configs/basic.yaml",
            "-w",
            "--stp-file",
            remage_stp_file,
            "--glm-file",
            f"{tmptestdir}/glm.lh5",
            "--hit-file",
            f"{tmptestdir}/hit.lh5",
            "--args",
            f"{test_file_dir}/configs/args.yaml",
        ]
    )

    hit1 = lh5.read("hit/det1", f"{tmptestdir}/hit.lh5").view_as("ak")
    assert set(hit1.fields) == {"t0", "t0_u", "evtid", "energy", "xloc"}
