# Real-MegaCam label-set score

Fixture: `benchmarks/trail_truth/megacam_real_labels.json` (schema weightmask.trail_truth.v1). Detectors: none, houghpeaks, streaks.
Coverage threshold 0.25 of a 4 px-wide labelled band.

| detector | trail recall | artefact entries | artefact FPs | FP rate | artefact pixel coverage | HDUs | seconds |
|---|---|---|---|---|---|---|---|
| none | n/a | 241 | 0 | 0.000 | 0.00000 | 89 | 0 |
| houghpeaks | n/a | 241 | 30 | 0.124 | 0.05856 | 89 | 282 |
| streaks | n/a | 241 | 30 | 0.124 | 0.06166 | 89 | 885 |

Trail recall is `n/a`: the fixture carries no confirmed real trail. The line that can be scored is the
false-positive rate on real clutter, which is what the audit's injected-trail suite cannot measure.

## Gate failures

- houghpeaks: artefact FP rate 0.124 > 0.100
- streaks: artefact FP rate 0.124 > 0.100
