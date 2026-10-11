"""Live progress: parsing Whisper's verbose lines and keeping the latest ones."""
import unittest

from npusubs.progress import Progress, SegmentTap


class SegmentTapTest(unittest.TestCase):
    def feed(self, text):
        got = []
        tap = SegmentTap(lambda s, e, t: got.append((s, e, t)))
        for chunk in (text[i:i + 7] for i in range(0, len(text), 7)):      # arrives in arbitrary pieces
            tap.write(chunk)
        return got

    def test_parses_minutes_and_hours_and_text_with_unicode(self):
        got = self.feed("[00:00.000 --> 00:05.120]  Hello there\n[01:02.500 --> 01:04.000] 日本語のテスト\n"
                        "[1:00:01.000 --> 1:00:03.500]  Late line\nnot a segment line\n")
        self.assertEqual(got, [(0.0, 5.12, "Hello there"), (62.5, 64.0, "日本語のテスト"), (3601.0, 3603.5, "Late line")])

    def test_a_line_is_only_reported_when_it_is_complete(self):
        got = []
        tap = SegmentTap(lambda s, e, t: got.append(t))
        tap.write("[00:00.000 --> 00:01.000]  half")
        self.assertEqual(got, [])
        tap.write(" a line\n")
        self.assertEqual(got, ["half a line"])


class ProgressTest(unittest.TestCase):
    def test_percent_recent_lines_and_time_mapping(self):
        p = Progress(100.0, to_orig=lambda t: t + 10)
        self.assertEqual(p.snapshot(), {"percent": 0, "segments": 0, "recent": []})
        for i in range(20):
            p.add(i * 5.0, i * 5.0 + 4, f"line {i}")
        p.add(95.0, 99.0, "")                                               # empty text is ignored
        snap = p.snapshot()
        self.assertEqual(snap["segments"], 20)
        self.assertEqual(len(snap["recent"]), Progress.RECENT)
        self.assertEqual(snap["recent"][-1], {"start": 105.0, "text": "line 19"})       # mapped back to the original timeline
        self.assertEqual(snap["percent"], 99)
        p.add(0, 500, "past the end")
        self.assertEqual(p.snapshot()["percent"], 100)


if __name__ == "__main__":
    unittest.main()
