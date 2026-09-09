"""Unit tests for pure functions in main.py — no external services required.

Run inside the backend container:
    docker compose exec backend python -m unittest test_main -v
Or locally with the backend deps installed:
    cd backend && python -m unittest test_main -v
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))

import main


class TestCleanMarkdown(unittest.TestCase):
    def test_strips_images(self):
        md = "Intro text\n\n![alt](http://x/img.png)\n\nMore text"
        out = main.clean_markdown(md)
        self.assertNotIn("img.png", out)
        self.assertIn("Intro text", out)
        self.assertIn("More text", out)

    def test_strips_link_only_lines(self):
        md = "Real content\n\n[Click here](https://example.com)\n\nMore content"
        out = main.clean_markdown(md)
        self.assertNotIn("Click here", out)
        self.assertIn("Real content", out)

    def test_strips_boilerplate(self):
        md = "Article body\n\nAdvertisement\n\nCredit: John Doe\n\nRelated: stuff"
        out = main.clean_markdown(md)
        self.assertNotIn("Advertisement", out)
        self.assertNotIn("Credit:", out)
        self.assertNotIn("Related:", out)
        self.assertIn("Article body", out)

    def test_strips_template_stubs(self):
        md = "Content\n\n{{ notfound }}\n\nMore"
        out = main.clean_markdown(md)
        self.assertNotIn("{{", out)

    def test_collapses_blank_runs(self):
        md = "Para one\n\n\n\n\n\nPara two"
        out = main.clean_markdown(md)
        self.assertNotIn("\n\n\n", out)

    def test_dedupes_repeated_paragraphs(self):
        md = "The Title Block\n\nSome unique content\n\nThe Title Block"
        out = main.clean_markdown(md)
        self.assertEqual(out.count("The Title Block"), 1)
        self.assertIn("Some unique content", out)

    def test_keeps_inline_links_in_sentences(self):
        md = "See [the docs](https://example.com) for details."
        out = main.clean_markdown(md)
        self.assertIn("the docs", out)


class TestChunkText(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(main.chunk_text("Hello world"), ["Hello world"])

    def test_splits_on_paragraph_boundaries(self):
        paras = [f"Paragraph {i} " + "x" * 400 for i in range(10)]
        chunks = main.chunk_text("\n\n".join(paras), size=1000, overlap=100)
        self.assertGreater(len(chunks), 1)

    def test_heading_prepended_to_chunks(self):
        text = "# Section One\n\nBody one.\n\n# Section Two\n\nBody two."
        chunks = main.chunk_text(text)
        self.assertTrue(chunks[0].startswith("Section One"))
        self.assertTrue(chunks[1].startswith("Section Two"))
        self.assertIn("Body two.", chunks[1])

    def test_oversized_paragraph_hard_split(self):
        chunks = main.chunk_text("y" * 4000, size=1500, overlap=200)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(len(c) <= 1500 for c in chunks))

    def test_overlap_carries_tail(self):
        p1 = "alpha " + "a" * 1400
        p2 = "omega " + "b" * 1400
        chunks = main.chunk_text(p1 + "\n\n" + p2, size=1500, overlap=200)
        self.assertEqual(len(chunks), 2)
        # second chunk begins with a tail of the first chunk's text
        self.assertIn("a", chunks[1][:200])

    def test_empty_text(self):
        self.assertEqual(main.chunk_text(""), [])
        self.assertEqual(main.chunk_text("\n\n\n"), [])


class TestSettings(unittest.TestCase):
    def test_defaults_present(self):
        for key in ("collection", "top_k", "score_threshold", "filter_key",
                    "filter_value", "generate", "rerank", "candidate_k",
                    "hnsw_ef", "exact"):
            self.assertIn(key, main.DEFAULT_SETTINGS)

    def test_load_merges_file_over_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "settings.json")
            with open(f, "w") as fh:
                json.dump({"top_k": 9, "collection": "test"}, fh)
            with patch.object(main, "SETTINGS_FILE", f):
                s = main.load_settings()
            self.assertEqual(s["top_k"], 9)
            self.assertEqual(s["collection"], "test")
            self.assertEqual(s["candidate_k"], 20)  # default preserved

    def test_load_missing_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(main, "SETTINGS_FILE", os.path.join(d, "nope.json")):
                self.assertEqual(main.load_settings(), main.DEFAULT_SETTINGS)

    def test_load_corrupt_file_returns_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "settings.json")
            with open(f, "w") as fh:
                fh.write("{not json")
            with patch.object(main, "SETTINGS_FILE", f):
                self.assertEqual(main.load_settings(), main.DEFAULT_SETTINGS)

    def test_save_and_reload_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "sub", "settings.json")
            with patch.object(main, "SETTINGS_FILE", f):
                main.save_settings({"top_k": 7})
                self.assertEqual(main.load_settings()["top_k"], 7)


if __name__ == "__main__":
    unittest.main()
