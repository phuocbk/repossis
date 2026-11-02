Independent AUTO evaluation branch: RM36 and PE-RLHF

Prepared only. No simulator run or checkpoint evaluation has been started.
No trainer imports this folder. No checkpoint or replay is written here by a trainer.
Q-estimator warm-up retains its original reward in PE_RLHF_SCANeR.

Evaluation metric:
  r_t = (d_t - d_previous) + 0.1 * speed_kmh / 80 + 20 * goal
  episode_return = sum(r_t), without discounting or normalization to 100.
  d is reference-route arc length in meters, not distance to goal or world x.
  Speed is the magnitude of velocity in km/h; reverse progress remains negative.
  Goal bonus is awarded only on the terminal goal step.
  No additional lane, collision, offroad or timeout penalty is in this formula.
  This metric alone does not measure lane discipline; report lane invasion separately.

Future AUTO runner integration:
  Use protocol.json identically for both algorithms.
  Set explicit checkpoint paths and hash checkpoints before/after each test.
  Use one shared reference polyline, home, goal, offroad rules and reset protocol.
  Retain the native observation/preprocessing of each trained actor.
  Deterministic actor actions only; disable training, human takeover and selector intervention.
  After reset create EpisodeReward(initial_route_progress_m).
  On each valid transition call add(next_progress_m, abs(speed_mps) * 3.6, terminal_reason).
  Terminate on goal/collision/offroad/40-second stall/1000-step horizon.
  Never score teleport/reset transitions. Exclude incomplete sensor-failure episodes.
  Save step-level raw telemetry and reward components, per-episode sums, success counts,
  checkpoint provenance and protocol in a new results/<algorithm>/<unique_run_id>/ directory.
  Abort if an output directory already exists; preserve prior results.

results/ is reserved for future evaluation output; it currently contains no measured results.
run_pe_rlhf_auto.py now runs PE-RLHF deterministic AUTO evaluation with an explicit checkpoint.
It reads the checkpoint's recorded config and writes only under results/PE_RLHF/eval_<timestamp>/.
The new metric is in evaluation_reward_episodes.json and evaluation_reward_summary.json.
Legacy episodes.json/summary.json are retained as native telemetry; their score is NOT this metric.
The checkpoint hash and learner update count are checked before/after the evaluation.
RM36 live launcher integration remains a future step.

RM36 threshold30 retained checkpoint sweep:
  python run_rm36_sweep.py --plan-only
  python run_rm36_sweep.py --suite <saved_RM36_suite_folder>
  Default branch: phuoc8_threshold30_episode_band30_v1; retained snapshots: 3000 and 4000 steps.
  Each checkpoint gets one deterministic AUTO episode with at most 200 steps.
  The evaluation is performed under the same ScanerAdapter, route, goal, reward and
  speed/timeout protocol used for the PE-RLHF sweep. Sensor failures abort without scoring.
  Native RM36 path100 observation and actor scaling are read from its training metadata.
  No filter is applied to AUTO inference; RM36 filtered HUMAN data during training.
  Only actor weights are loaded; no learner, optimizer or replay is initialized for RM36.
  Completed tasks are skipped on resume. Raw telemetry and provenance stay in this folder.
  Four real-result plots are generated automatically when all tasks complete.
  Original RM36 20-episode evaluations used another score/horizon and are not reused here.
  RM36 inherited a straight-driving checkpoint, whereas PE-RLHF online actor started fresh;
  report this initialization difference when interpreting training speed comparisons.

PE-RLHF checkpoint sweep:
  python checkpoint_sweep.py --every 1 --tests 1 --max-episode-steps 200
  Current defaults: every retained checkpoint, one episode each, 200-step horizon.
  At step 200 an unfinished episode ends as time_limit, receives no goal bonus,
  saves its measured return and advances to the next checkpoint after reset.
  --max-episode-steps changes evaluation only; training configurations are untouched.
  Existing suite folders keep their original saved horizon. Use a new suite to change it.
  Default source: phuoc8_pe_rlhf_online4000_noise_retest_v3.
  --plan-only creates a manifest without connecting to SCANeR.
  --suite <existing_suite_directory> runs/resumes a saved plan.
  --suite <existing_suite_directory> --plot-only refreshes plots without SCANeR.
  Each sweep gets a unique results/sweeps/<timestamp>/ folder.
  One SDK connection is reused across checkpoints and episodes.
  Load/Play phuoc8 when first prompted; keep the engine on and physical controls neutral.
  No training, console takeover, physics intervention or Q-estimator selection runs.
  Models and their training files are read only.
  Training-step x-axis uses the checkpoint's recorded online step counter (not updates).
  Warm-up steps are excluded from this axis, and must be reported separately in a paper.
  Checkpoints are sampled at multiples of --every; final retained checkpoint is included.
  Duplicate saved snapshots at the same step become one task.
  Only available checkpoints can be evaluated; no missing model weights are reconstructed.
  Completed checkpoint results are reused on resume after their checkpoint hash is checked.
  An interrupted checkpoint is rerun in full; old partial sessions are retained but not pooled.
  Sensor/telemetry failures stop the sweep rather than being scored as policy failures.
  checkpoint_metrics.csv, checkpoint_curves.png and checkpoint_curves.pdf update per checkpoint.
  Plot columns: mean episode return, goal success rate, offroad rate.
  Return bands are approximate 95% CIs for the mean; rates use Wilson 95% CIs.
  These bands describe evaluation variation from ONE trained seed, not multi-seed training variance.
  Legacy episodes.json score is not used for these curves.
  --mock writes explicitly synthetic checks under results/mock_sweeps/, separate from real results.

Local checks:
  python -m unittest discover -s tests -v
