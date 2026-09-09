import unittest

from drror_vllm_ascend.prepare import selection_method


class SelectionProvenanceTests(unittest.TestCase):
    def test_official_raw_is_the_only_paper_exact_selector(self):
        method, paper_exact = selection_method("official-raw")
        self.assertEqual(method, "DRRQR (paper-faithful Strong RRQR selector)")
        self.assertTrue(paper_exact)

        for objective in ("cosine-kernel", "energy-kernel"):
            method, paper_exact = selection_method(objective)
            self.assertIn("Experimental", method)
            self.assertFalse(paper_exact)

    def test_unknown_selector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown selection objective"):
            selection_method("not-a-selector")


if __name__ == "__main__":
    unittest.main()
