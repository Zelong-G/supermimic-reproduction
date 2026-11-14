"""Check matching boundaries and corrupt-data rejection, without loading simulation."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from analyze import (DEFAULT_FULL, DEFAULT_SEG, aggregate, category, load_results,
                     match_segments, split_segment)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = load_results(DEFAULT_FULL)
        cls.seg = load_results(DEFAULT_SEG)

    def test_actual_data_conservation(self):
        rows, segments = match_segments(self.full['clips'], self.seg['clips'])
        self.assertEqual(len(rows), 40)
        self.assertEqual(len(segments), 882)
        self.assertEqual(sum(r['segment_failures'] for r in rows), 37)
        self.assertEqual(sum(r['covered_frames'] + r['omitted_tail_frames'] for r in rows),
                         sum(c['frames'] for c in self.full['clips']))
        self.assertEqual(sum(r['omitted_tail_frames'] for r in rows), 3309)
        self.assertEqual(sum(r['segment_overflow_steps'] for r in rows), 18)

    def test_missing_duplicate_or_unmatched_segment_rejected(self):
        clips = self.seg['clips']
        for broken in [clips[1:], clips + [clips[0]],
                       [{**clips[0], 'name': 'unknown__seg000'}] + clips[1:]]:
            with self.subTest(size=len(broken)):
                with self.assertRaises(ValueError):
                    match_segments(self.full['clips'], broken)

    def test_tail_boundary(self):
        template = self.full['clips'][0]
        seg = self.seg['clips'][0]
        for frames, lengths in [(749, [500]), (750, [500, 250]), (1000, [500, 500])]:
            full = [{**template, 'name': 'walk1_subject1', 'frames': frames}]
            parts = [{**seg, 'name': f'walk1_subject1__seg{i:03}', 'frames': n}
                     for i, n in enumerate(lengths)]
            rows, _ = match_segments(full, parts)
            self.assertEqual(rows[0]['omitted_tail_frames'], frames-sum(lengths))

    def test_bad_summary_rejected(self):
        data = copy.deepcopy(self.full)
        data['summary']['failed'] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'corrupt.json'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'stored summary differs'):
                load_results(path)

    def test_weighting_uses_complete_reference_frames(self):
        a, b = copy.deepcopy(self.full['clips'][:2])
        a.update(frames=100, norm_return=.2)
        b.update(frames=300, norm_return=.8)
        stats = aggregate([a,b])
        self.assertAlmostEqual(stats['mean_norm_return'], .5)
        self.assertAlmostEqual(stats['step_weighted_norm_return'], .65)

    def test_categories_and_names(self):
        self.assertEqual(category('fallAndGetUp2_subject3'), 'fallAndGetUp')
        self.assertEqual(split_segment('dance1_subject2__seg003'), ('dance1_subject2', 3))
        with self.assertRaises(ValueError):
            split_segment('dance1_subject2_seg003')
        with self.assertRaises(ValueError):
            category('side_roll_001')


if __name__ == '__main__':
    unittest.main()
