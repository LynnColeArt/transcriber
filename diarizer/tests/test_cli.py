import tempfile
import unittest
from pathlib import Path

from transcriber_diarizer.cli import annotation_segments, select_device, write_json_atomic


class Turn:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class Annotation:
    def itertracks(self, yield_label: bool = False):
        assert yield_label
        yield Turn(2.0, 3.5), None, "SPEAKER_01"
        yield Turn(0.1, 1.2), None, "SPEAKER_00"


class TorchWithoutCUDA:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False


class CLITest(unittest.TestCase):
    def test_annotation_segments_are_stable_and_sorted(self) -> None:
        self.assertEqual(
            annotation_segments(Annotation()),
            [
                {"start": 0.1, "end": 1.2, "speaker": "SPEAKER_00"},
                {"start": 2.0, "end": 3.5, "speaker": "SPEAKER_01"},
            ],
        )

    def test_auto_device_falls_back_to_cpu(self) -> None:
        self.assertEqual(select_device("auto", TorchWithoutCUDA), "cpu")
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            select_device("cuda", TorchWithoutCUDA)

    def test_atomic_json_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_json_atomic(path, {"speakers": ["SPEAKER_00"]})
            self.assertIn('"SPEAKER_00"', path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
