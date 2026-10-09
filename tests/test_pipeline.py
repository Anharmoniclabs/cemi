import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from cemi_pipeline.cli import main


class PipelineTests(unittest.TestCase):
    def test_shot(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "shot.json"
            main(["shot", "--character", "JC", "--shot-id", "ep01_sh01",
                  "--scene", "Bronx rooftop at dusk", "--out", str(out)])
            data = json.loads(out.read_text())
            assert data["character_id"] == "JC"
            assert "wall writing" in data["negative_prompt"]
            assert "cemistyle" in data["prompt"]

    def test_dataset_enforces_approval(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            Image.new("RGB", (600, 600), (60, 40, 20)).save(path / "art.png")
            manifest = path / "manifest.json"
            example = {"file": "art.png", "approved": False, "origin": "author_original",
                       "caption": "full color hand drawn cinematic character"}
            manifest.write_text(json.dumps([example]))
            with self.assertRaises(SystemExit):
                main(["prepare", "--manifest", str(manifest), "--out", str(path / "train")])
            example["approved"] = True
            manifest.write_text(json.dumps([example]))
            main(["prepare", "--manifest", str(manifest), "--out", str(path / "train")])
            assert (path / "train" / "metadata.jsonl").exists()

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            manifest = path / "manifest.json"
            manifest.write_text(json.dumps([
                {"file": "../wrong.png", "approved": True, "origin": "author_original",
                 "caption": "mature original full color portrait art"}
            ]))
            with self.assertRaises(SystemExit):
                main(["prepare", "--manifest", str(manifest), "--out", str(path / "train")])


if __name__ == "__main__":
    unittest.main()
