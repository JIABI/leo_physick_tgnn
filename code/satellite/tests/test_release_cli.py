from __future__ import annotations

import json
from pathlib import Path

import yaml

from leo_pg.paper.release_cli import (
    DEFAULT_AUTHOR,
    DEFAULT_CONFIG,
    DEFAULT_STUDIES,
    main as release_main,
)


def test_plan_all_dispatches_every_registered_study(tmp_path, capsys) -> None:
    """Keep the public ``--select all`` command executable as studies evolve."""

    author = yaml.safe_load(DEFAULT_AUTHOR.read_text(encoding="utf-8"))
    author["runs"] = author["runs"][:1]
    author["held_out_panels"] = {
        "run01": author["held_out_panels"]["run01"]
    }
    author_path = tmp_path / "author.yaml"
    author_path.write_text(
        yaml.safe_dump(author, sort_keys=False), encoding="utf-8"
    )
    output_root = tmp_path / "plan"

    release_main(
        [
            "plan",
            "--config",
            str(DEFAULT_CONFIG),
            "--studies",
            str(DEFAULT_STUDIES),
            "--author",
            str(author_path),
            "--select",
            "all",
            "--out",
            str(output_root),
            "--device",
            "cpu",
        ]
    )

    console = json.loads(capsys.readouterr().out)
    plan = json.loads(
        (output_root / "satellite_plan.json").read_text(encoding="utf-8")
    )
    registered = set(
        yaml.safe_load(DEFAULT_STUDIES.read_text(encoding="utf-8"))["studies"]
    )
    assert set(plan["selected_studies"]) == registered
    assert "SNAPSHOT-WEIGHT-CONTRACT" in plan["selected_studies"]
    assert console["task_count"] == len(plan["tasks"])
    assert console["task_count"] > 0
    assert all(Path(task["outputs"][0]).is_absolute() for task in plan["tasks"])
