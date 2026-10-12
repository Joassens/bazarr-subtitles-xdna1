"""The host agent reports RAM and, when the amdgpu counter exists, the GPU's recent load."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import hostmem_agent  # noqa: E402


class GpuSamplerTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "gpu_busy_percent")

    def set(self, v):
        with open(self.path, "w") as f:
            f.write(f"{v}\n")

    def test_window_max_avg_and_last(self):
        s = hostmem_agent.GpuSampler([self.path])
        for t, v in ((100, 0), (101, 40), (102, 20), (103, 0)):
            self.set(v)
            s.sample(now=t)
        snap = s.snapshot(now=103)
        self.assertEqual(snap["gpu_busy_pct"], 0)
        self.assertEqual(snap["gpu_busy_max_10s"], 40)
        self.assertEqual(snap["gpu_busy_avg_10s"], 15.0)

    def test_old_samples_leave_the_window(self):
        s = hostmem_agent.GpuSampler([self.path])
        self.set(90)
        s.sample(now=100)
        self.set(0)
        s.sample(now=112)
        self.assertEqual(s.snapshot(now=112)["gpu_busy_max_10s"], 0)

    def test_no_gpu_counter_means_no_gpu_numbers(self):
        s = hostmem_agent.GpuSampler([os.path.join(self.dir, "missing")])
        s.sample()
        self.assertEqual(s.snapshot(), {})

    def test_the_busiest_gpu_counts(self):
        other = os.path.join(self.dir, "second")
        with open(other, "w") as f:
            f.write("55\n")
        self.set(5)
        self.assertEqual(hostmem_agent.GpuSampler([self.path, other]).read(), 55)


if __name__ == "__main__":
    unittest.main()
