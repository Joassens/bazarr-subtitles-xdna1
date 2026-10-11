"""Unit tests for the pure-Python parts (no NPU, no model downloads needed).

Run:  python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from npusubs import filters, subtitles
from npusubs.diarize import assign_speakers
from npusubs.vad import Piece, TimeMap, Vad


def words(spec, speaker=None):
    """spec: list of (text, start, end)"""
    return [{"word": t, "start": s, "end": e, "speaker": speaker} for t, s, e in spec]


class TimeMapTest(unittest.TestCase):
    def setUp(self):
        # speech 5..8 s and 12..15 s; condensed: piece0 at 0.0, piece1 at 3.3 (3 s speech + 0.3 s gap)
        self.tm = TimeMap([Piece(5, 8, 0.0), Piece(12, 15, 3.3)])

    def test_inside_piece(self):
        self.assertAlmostEqual(self.tm.to_orig(1.0), 6.0)
        self.assertAlmostEqual(self.tm.to_orig(4.3), 13.0)

    def test_start_in_removed_gap_snaps_forward(self):
        self.assertAlmostEqual(self.tm.to_orig(3.1), 12.0)

    def test_end_in_removed_gap_snaps_back(self):
        self.assertAlmostEqual(self.tm.to_orig(3.1, end=True), 8.0)

    def test_after_last_piece(self):
        self.assertAlmostEqual(self.tm.to_orig(99.0, end=True), 15.0)

    def test_identity_without_pieces(self):
        self.assertEqual(TimeMap([]).to_orig(4.2), 4.2)


class MusicRegionTest(unittest.TestCase):
    def test_sustained_loud_gap_is_music_but_short_pause_is_not(self):
        sr = 16000
        rng = np.random.default_rng(0)
        loud = lambda s: (rng.standard_normal(int(s * sr)) * 0.1).astype(np.float32)
        quiet = lambda s: np.zeros(int(s * sr), dtype=np.float32)
        audio = np.concatenate([loud(30), quiet(2), loud(5), quiet(40)])   # 30 s "song", short blip, long silence
        v = Vad(music=True)
        regions = v.music_regions(audio, speech=[(32.0, 37.0)])
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0][0], 0.0, delta=0.3)
        self.assertGreater(regions[0][1], 29.0)

    def test_disabled(self):
        self.assertEqual(Vad(music=False).music_regions(np.ones(16000 * 30, np.float32), []), [])


class SubtitlesTest(unittest.TestCase):
    def test_split_on_gap(self):
        segs = [{"text": "hello world", "start": 0, "end": 1, "words": words([(" hello", 0, 0.5), (" world", 0.5, 1.0), (" again", 3.0, 3.5)])}]
        cues = subtitles.build_cues(segs)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[1].text, "again")

    def test_two_line_split_is_balanced(self):
        ws = [(" " + w, i * 0.3, i * 0.3 + 0.3) for i, w in enumerate(
            "this is a rather long sentence that must wrap onto two lines nicely".split())]
        cues = subtitles.build_cues([{"text": "x", "start": 0, "end": 5, "words": words(ws)}],
                                    subtitles.Options(max_line=30, gap_split=2.0, speaker_style="none"))
        self.assertTrue(all(subtitles.width(line) <= 30 for c in cues for line in c.text.split("\n")))
        self.assertTrue(any("\n" in c.text for c in cues))

    def test_cjk_counts_double_width(self):
        self.assertEqual(subtitles.width("日本語"), 6)
        self.assertEqual(subtitles.width("abc"), 3)

    def test_speaker_change_splits_and_dash_merges(self):
        w = words([(" Hi", 0.0, 0.4)], "A") + words([(" Hey", 0.6, 1.0)], "B")
        cues = subtitles.build_cues([{"text": "x", "start": 0, "end": 1, "words": w}], subtitles.Options(speaker_style="dash"))
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].text, "- Hi\n- Hey")

    def test_label_style(self):
        w = words([(" Hi", 0.0, 0.4)], "SPEAKER_00")
        cues = subtitles.build_cues([{"text": "x", "start": 0, "end": 1, "words": w}], subtitles.Options(speaker_style="label"))
        self.assertTrue(cues[0].text.startswith("[SPEAKER_00]"))

    def test_srt_and_vtt_format(self):
        cues = [subtitles.Cue(1.5, 3.25, "Line")]
        self.assertIn("00:00:01,500 --> 00:00:03,250", subtitles.to_srt(cues))
        self.assertTrue(subtitles.to_vtt(cues).startswith("WEBVTT"))
        self.assertIn("00:00:01.500 --> 00:00:03.250", subtitles.to_vtt(cues))

    def test_min_duration_does_not_overlap_next(self):
        cues = subtitles.build_cues([{"text": "a b", "start": 0, "end": 1, "words": words([(" a", 0.0, 0.1), (" b", 0.7, 0.8)])}],
                                    subtitles.Options(gap_split=0.3, speaker_style="none"))
        for a, b in zip(cues, cues[1:]):
            self.assertLessEqual(a.end, b.start)


class FiltersTest(unittest.TestCase):
    def seg(self, text, s=0.0, e=1.0):
        return {"text": text, "start": s, "end": e, "words": []}

    def test_drop_empty(self):
        self.assertEqual([s["text"] for s in filters.drop_empty([self.seg("."), self.seg("hi")])], ["hi"])

    def test_collapse_loop(self):
        segs = [self.seg("same line", i, i + 1) for i in range(8)] + [self.seg("real", 9, 10)]
        out = filters.collapse_loops(segs)
        self.assertEqual([s["text"] for s in out], ["same line", "real"])
        self.assertEqual(out[0]["end"], 8)

    def test_alternating_loop(self):
        segs = [self.seg("aaaa" if i % 2 == 0 else "bbbb", i, i + 1) for i in range(10)]
        self.assertEqual(len(filters.collapse_loops(segs)), 2)

    def test_recurring_lines_dropped_but_short_ones_kept(self):
        segs = [self.seg("thanks for watching", i * 10, i * 10 + 1) for i in range(3)] + [self.seg("yes", 40, 41)] * 3
        out = [s["text"] for s in filters.drop_recurring(segs)]
        self.assertEqual(out, ["yes", "yes", "yes"])

    def test_phantom_start(self):
        segs = [self.seg("hello there", 0.1, 0.6), self.seg("x", 3, 4), self.seg("hello there", 5, 6)]
        self.assertEqual(len(filters.drop_phantom_start(segs)), 2)


class DiarizeAssignTest(unittest.TestCase):
    def test_overlap_wins_and_first_seen_order(self):
        segs = [{"text": "a b", "start": 0, "end": 4, "words": words([(" a", 0.0, 1.0), (" b", 2.5, 3.5)])}]
        turns = [(0.0, 2.0, "spk_z"), (2.0, 4.0, "spk_q")]
        out = assign_speakers(segs, turns)
        self.assertEqual(out[0]["words"][0]["speaker"], "SPEAKER_00")
        self.assertEqual(out[0]["words"][1]["speaker"], "SPEAKER_01")

    def test_no_turns_is_noop(self):
        segs = [{"text": "a", "start": 0, "end": 1, "words": words([(" a", 0, 1)])}]
        self.assertIs(assign_speakers(segs, []), segs)


if __name__ == "__main__":
    unittest.main()
