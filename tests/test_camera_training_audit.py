"""Source-pool assignment identity is distinct from successful reconstruction."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_camera_training_sensitivity import compare_assignments, proposal_assignment
from complete_camera_training_audit import require_same_json_audit


def fixture():
    frames = [dict(pool=dict(candidates=[dict(row_matches=[[0, 0, 0.]])]),
                   observations=dict(rows=[dict(candidates=[dict(center_x=10.)])])) for _ in range(3)]
    ids = ['left', 'middle', 'right']
    proposal = dict(ordinal=0, hypothesis=dict(matches=[dict(candidate_index=0)] * 3, supporting_views=[0, 1, 2]),
                    finite=dict(state='accepted', candidate_selection=[dict(view_id=view, candidate_index=0, row_matches=[[0, 0]]) for view in ids]))
    return proposal, ids, frames


class TrainingAuditTests(unittest.TestCase):
    def test_saved_source_rows_keep_meaning_but_changed_evidence_is_rejected(self):
        live = dict(source_rows=[(0, 2, 1)], receipts={"inference.json": "fixed-hash"})
        saved = json.loads(json.dumps(live))
        self.assertNotEqual(live, saved)
        require_same_json_audit(saved, live)
        changed = copy.deepcopy(saved)
        changed["source_rows"][0][2] = 3
        with self.assertRaisesRegex(ValueError, "changed"):
            require_same_json_audit(saved, changed)
        changed = copy.deepcopy(saved)
        changed["receipts"]["inference.json"] = "another-hash"
        with self.assertRaisesRegex(ValueError, "changed"):
            require_same_json_audit(saved, changed)

    def test_source_pool_rows_are_resolved_and_detached_rows_rejected(self):
        proposal, ids, frames = fixture()
        result = proposal_assignment(proposal, ids, frames, 'a' * 64, 8)
        self.assertEqual(result['source_rows'], [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        self.assertEqual(result['per_view_row_count'], [1, 1, 1])
        changed = copy.deepcopy(proposal)
        changed['finite']['candidate_selection'][0]['row_matches'].append([0, 0])
        with self.assertRaisesRegex(ValueError, 'repeated'):
            proposal_assignment(changed, ids, frames, 'a' * 64, 8)
        changed['finite']['candidate_selection'][0]['row_matches'] = [[1, 0]]
        with self.assertRaisesRegex(ValueError, 'detached'):
            proposal_assignment(changed, ids, frames, 'a' * 64, 8)

    def test_jaccard_uses_row_identity_and_rejected_or_empty_is_not_agreement(self):
        proposal, ids, frames = fixture()
        selected = proposal_assignment(proposal, ids, frames, 'a' * 64, 8)
        left = dict(state='accepted', selected=selected)
        right = copy.deepcopy(left)
        right['selected']['source_rows'] = [(0, 0, 0), (1, 0, 0)]
        self.assertAlmostEqual(compare_assignments(left, right)['support_row_jaccard'], 2 / 3)
        right['selected']['source_rows'] = []
        self.assertFalse(compare_assignments(left, right)['comparable'])
        rejected = dict(state='unresolved', selected=None)
        self.assertIsNone(compare_assignments(rejected, rejected)['support_row_jaccard'])
        right = copy.deepcopy(left)
        for row in right['selected']['hypothesis_assignments']:
            row['pool_sha256'] = 'b' * 64
        self.assertEqual(compare_assignments(left, right)['reason'], 'different_source_pixel_pools')


if __name__ == '__main__':
    unittest.main()
