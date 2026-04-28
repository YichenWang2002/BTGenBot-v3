import csv
import json
from pathlib import Path

from src.dataset.profile import analyze_xml, main


def test_analyze_xml_uses_regex_fallback_for_malformed_xml():
    malformed = """
<root>
  <!--------------------------------------->
  <BehaviorTree ID="MainTree">
    <Action ID="Pick"/>
    <Condition ID="Bottle_found"/>
  </BehaviorTree>
</root>
"""

    result = analyze_xml(malformed)

    assert result["parse_error"]
    assert result["xml_node_count"] == 4
    assert result["xml_tags"] == ["Action", "BehaviorTree", "Condition", "root"]
    assert result["action_condition_names"] == ["Bottle_found", "Pick"]


def test_cli_generates_outputs_and_prints_summary(tmp_path, monkeypatch, capsys):
    dataset_path = tmp_path / "sample_dataset.json"
    out_dir = tmp_path / "derived"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "instruction": "inst",
                    "input": "navigate to a goal",
                    "output": (
                        "<root><BehaviorTree ID='MainTree'>"
                        "<Sequence><ComputePathToPose goal='{goal}' path='{path}'/>"
                        "<FollowPath path='{path}'/></Sequence>"
                        "</BehaviorTree></root>"
                    ),
                },
                {
                    "instruction": "inst",
                    "input": "recover and pick",
                    "output": (
                        "<root><BehaviorTree ID='MainTree'>"
                        "<RecoveryNode><Action ID='Pick'/><Wait delay='1'/>"
                        "</RecoveryNode></BehaviorTree></root>"
                    ),
                },
            ]
        ),
        encoding="utf-8",
    )

    assert main(["--dataset", str(dataset_path), "--out", str(out_dir)]) == 0
    stdout = capsys.readouterr().out

    assert "Total samples: 2" in stdout
    assert "navigation=1" in stdout
    assert "recovery=1" in stdout
    assert "operation=1" in stdout
    assert "pick_place=1" in stdout
    assert "RecoveryNode sample indexes: [1]" in stdout
    assert "Pick/place sample indexes: [1]" in stdout

    profile_json = out_dir / "bt_dataset_profile.json"
    index_csv = out_dir / "bt_dataset_index.csv"
    candidates_json = out_dir / "scenario_candidates.json"
    assert profile_json.exists()
    assert index_csv.exists()
    assert candidates_json.exists()

    profile = json.loads(profile_json.read_text(encoding="utf-8"))
    assert profile["sample_count"] == 2
    assert profile["counts"]["navigation"] == 1
    assert profile["samples"][1]["action_condition_names"] == ["Pick", "Wait"]

    with index_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["index"] == "0"
    assert rows[1]["has_recovery_node"] == "True"

    candidates = json.loads(candidates_json.read_text(encoding="utf-8"))
    assert [row["index"] for row in candidates["pick_place"]] == [1]


def test_cli_falls_back_from_data_path_to_dataset_path(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    fallback_path = Path("dataset/bt_dataset.json")
    fallback_path.parent.mkdir()
    fallback_path.write_text(
        json.dumps(
            [
                {
                    "instruction": "inst",
                    "input": "place object",
                    "output": "<root><BehaviorTree><Action ID='Place'/></BehaviorTree></root>",
                }
            ]
        ),
        encoding="utf-8",
    )

    assert main(["--dataset", "data/bt_dataset.json", "--out", "data/derived"]) == 0
    captured = capsys.readouterr()

    assert "falling back to dataset/bt_dataset.json" in captured.err
    assert "Total samples: 1" in captured.out
    assert Path("data/derived/bt_dataset_profile.json").exists()
