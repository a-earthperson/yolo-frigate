import os
import unittest

from yolo_frigate.label import parse_classes

TEST_DIR = os.path.dirname(__file__)


class TestParseLabels(unittest.TestCase):
    def test_parse_labels_yaml(self):
        classes = parse_classes(os.path.join(TEST_DIR, "labelmap.yml"))
        self.assertEqual(classes, ["Label One", "Label Two", "Label Three"])

    def test_parse_labels_text(self):
        classes = parse_classes(os.path.join(TEST_DIR, "labelmap.txt"))
        self.assertEqual(classes, ["Label One", "Label Two", "Label Three"])


if __name__ == "__main__":
    unittest.main()
