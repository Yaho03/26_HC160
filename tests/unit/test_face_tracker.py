import unittest

from src.face_auth.inference.face_tracker import SingleFaceTracker


class SingleFaceTrackerTest(unittest.TestCase):
    def test_overlapping_boxes_keep_track(self):
        tracker = SingleFaceTracker()
        first = tracker.update((0, 0, 100, 100))
        second = tracker.update((5, 5, 105, 105))
        self.assertEqual(first.track_id, second.track_id)
        self.assertFalse(second.switched)

    def test_distant_box_starts_new_track(self):
        tracker = SingleFaceTracker()
        first = tracker.update((0, 0, 50, 50))
        second = tracker.update((100, 100, 150, 150))
        self.assertNotEqual(first.track_id, second.track_id)
        self.assertTrue(second.switched)


if __name__ == "__main__":
    unittest.main()
