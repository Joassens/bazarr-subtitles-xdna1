"""The job gate: one request at a time, in order, with a bounded wait."""
import threading
import time
import unittest

from npusubs.jobs import JobGate


class JobGateTest(unittest.TestCase):
    def test_never_more_than_one_job_runs_at_a_time(self):
        gate, state, errors = JobGate(1), {"now": 0, "max": 0}, []
        lock = threading.Lock()

        def job():
            try:
                with gate.slot(10):
                    with lock:
                        state["now"] += 1
                        state["max"] = max(state["max"], state["now"])
                    time.sleep(0.05)
                    with lock:
                        state["now"] -= 1
            except Exception as exc:                  # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=job) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(errors, [])
        self.assertEqual(state["max"], 1)
        self.assertEqual((gate.running, gate.waiting), (0, 0))

    def test_jobs_run_in_the_order_they_arrived(self):
        gate, order = JobGate(1), []
        hold = threading.Event()

        def first():
            with gate.slot(10):
                hold.wait(5)

        def later(n):
            with gate.slot(10):
                order.append(n)
        t0 = threading.Thread(target=first)
        t0.start()
        time.sleep(0.1)
        threads = []
        for n in range(5):
            t = threading.Thread(target=later, args=(n,))
            t.start()
            threads.append(t)
            time.sleep(0.05)                          # arrival order is 0..4
        self.assertEqual(gate.waiting, 5)
        hold.set()
        [t.join() for t in [t0] + threads]
        self.assertEqual(order, [0, 1, 2, 3, 4])

    def test_waiting_too_long_gives_up_and_leaves_the_queue_clean(self):
        gate = JobGate(1)
        started, done = threading.Event(), threading.Event()

        def holder():
            with gate.slot(10):
                started.set()
                done.wait(5)
        t = threading.Thread(target=holder)
        t.start()
        started.wait(5)
        with self.assertRaises(TimeoutError):
            with gate.slot(0.2):
                pass
        self.assertEqual(gate.waiting, 0)
        done.set()
        t.join()
        with gate.slot(1):                            # and the gate still works
            self.assertEqual(gate.running, 1)

    def test_a_failing_job_releases_its_slot(self):
        gate = JobGate(1)
        with self.assertRaises(ValueError):
            with gate.slot(1):
                raise ValueError("boom")
        with gate.slot(1):
            pass

    def test_more_slots_allow_that_many_in_parallel(self):
        gate, inside = JobGate(2), []
        both = threading.Barrier(2, timeout=5)

        def job():
            with gate.slot(5):
                both.wait()                            # only passes when two jobs are inside at once
                inside.append(1)
        threads = [threading.Thread(target=job) for _ in range(2)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(len(inside), 2)


if __name__ == "__main__":
    unittest.main()
