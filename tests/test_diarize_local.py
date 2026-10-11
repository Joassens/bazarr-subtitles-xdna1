"""Token-free diarization: clustering, turns and the engine router (synthetic data, no model needed)."""
import unittest

import numpy as np

from npusubs import diarize_local as dl

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except Exception:                                   # noqa: BLE001
    HAVE_SCIPY = False


def voices(counts, noise=0.05, seed=1):
    """Embeddings of len(counts) distinct 'speakers' in a block, in order."""
    rng = np.random.default_rng(seed)
    centres = np.linalg.qr(rng.normal(size=(64, len(counts))))[0].T          # orthogonal directions
    rows, truth = [], []
    for i, n in enumerate(counts):
        rows.append(centres[i] + noise * rng.normal(size=(n, 64)))
        truth += [i] * n
    return dl._l2(np.concatenate(rows)), np.array(truth)


class HelpersTest(unittest.TestCase):
    def test_smooth_removes_a_one_window_flicker(self):
        lab = np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(list(dl.smooth_labels(lab)), [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    def test_turns_own_the_stretch_around_each_window_and_cover_the_whole_audio(self):
        starts = np.array([0.0, 0.5, 1.0, 1.5])
        turns = dl.labels_to_turns(np.array([0, 0, 1, 1]), starts, window=1.5, hop=0.5, duration=3.0)
        self.assertEqual([t[2] for t in turns], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(turns[0][0], 0.0)
        self.assertEqual(turns[-1][1], 3.0)
        self.assertAlmostEqual(turns[0][1], turns[1][0])                      # no gap between the turns

    def test_small_clusters_are_merged_into_the_nearest_big_one(self):
        emb, truth = voices([20, 20])
        stray = dl._l2(emb[0] + 0.05 * np.ones(64))[None, :]
        emb = np.concatenate([emb, stray])
        labels = np.concatenate([truth, [2]])
        merged = dl.merge_small_clusters(emb, labels, min_windows=3)
        self.assertEqual(len(set(merged)), 2)
        self.assertEqual(merged[-1], merged[0])


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class ClusterTest(unittest.TestCase):
    def test_finds_the_number_of_speakers(self):
        for counts in ([30], [30, 25], [30, 25, 20]):
            emb, _ = voices(counts)
            labels = dl.cluster_embeddings(emb, 0.8, None, None, min_windows=4)
            self.assertEqual(len(set(labels)), len(counts), counts)

    def test_labels_follow_the_voices(self):
        emb, truth = voices([30, 25, 20])
        labels = dl.cluster_embeddings(emb, 0.8, None, None, min_windows=4)
        for t in range(3):
            self.assertEqual(len(set(labels[truth == t])), 1)

    def test_min_and_max_speakers_win_over_the_threshold(self):
        emb, _ = voices([30, 25, 20])
        self.assertEqual(len(set(dl.cluster_embeddings(emb, 0.8, None, 2, 4))), 2)
        emb, _ = voices([30])
        self.assertGreaterEqual(len(set(dl.cluster_embeddings(emb, 0.8, 2, None, 1))), 2)

    def test_tiny_input(self):
        self.assertEqual(list(dl.cluster_embeddings(np.ones((1, 8)), 0.8, None, None, 1)), [0])


class RouterTest(unittest.TestCase):
    class Fake:
        def __init__(self, ok):
            self.available, self.status = ok, "ready" if ok else "no token"

    def router(self, py_ok, local_ok=True):
        return dl.DiarizerRouter(self.Fake(py_ok), self.Fake(local_ok))

    def test_auto_prefers_pyannote_and_falls_back_to_the_token_free_engine(self):
        r = self.router(True)
        self.assertIs(r.pick("auto"), r.pyannote)
        r = self.router(False)
        self.assertIs(r.pick("auto"), r.local)
        self.assertEqual(r.engine_name("auto"), "embedding")

    def test_explicit_engine_is_respected(self):
        r = self.router(True)
        self.assertIs(r.pick("embedding"), r.local)
        self.assertIs(r.pick("pyannote"), r.pyannote)
        r = self.router(False)
        self.assertIs(r.pick("pyannote"), r.pyannote)          # unavailable: the caller then skips speakers


if __name__ == "__main__":
    unittest.main()
