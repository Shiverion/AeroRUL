export interface ModelInfo {
  name: string;
  tagline: string;
  grounding: string;
  limitations: string;
}

export const MODEL_INFO: Record<string, ModelInfo> = {
  xgboost: {
    name: "XGBoost",
    tagline: "Gradient-boosted trees over engineered rolling-window features",
    grounding:
      "The standard first move for tabular regression — fast to train, a strong baseline, " +
      "and every other model here is judged against it. Rolling-window statistics (5/10/20-" +
      "cycle mean and standard deviation) let it see recent trend without needing a " +
      "sequential architecture at all.",
    limitations:
      "No native sense of time — it only sees the engineered snapshot at the current cycle, " +
      "not the shape of the trajectory leading up to it. On the harder, 6-condition subsets " +
      "(FD002/FD004) that gap barely matters: XGBoost is essentially tied with the deep " +
      "models there, or wins outright.",
  },
  lstm: {
    name: "LSTM",
    tagline: "Recurrent network reading a 30-cycle raw sensor window",
    grounding:
      "Learns its own temporal features directly from raw, normalized sensor values instead " +
      "of hand-crafted rolling stats — a natural fit for a sequence problem, carrying a " +
      "hidden state forward cycle by cycle.",
    limitations:
      "Processes the window strictly step by step, which makes it slower to train than TCN " +
      "or Transformer and, on this dataset (CPU-trained, 15 epochs with early stopping), " +
      "the least consistent of the three sequence models — it wins on FD004 but trails on " +
      "the others.",
  },
  tcn: {
    name: "TCN",
    tagline: "Dilated causal convolutions, exponentially growing receptive field",
    grounding:
      "No recurrence means it's fully parallelizable during training, and the causal, " +
      "backward-only receptive field matches how a live prediction actually has to work — " +
      "never looking at cycles that haven't happened yet.",
    limitations:
      "Its receptive field is fixed by architecture (window length and dilation depth), " +
      "unlike an LSTM's in-principle unbounded memory. It's the strongest performer on " +
      "FD002, the hardest subset here, but falls behind on FD004.",
  },
  transformer: {
    name: "Transformer",
    tagline: "Self-attention encoder, attends to any cycle in the window directly",
    grounding:
      "No fixed receptive field and no step-by-step bottleneck — attention weighs every " +
      "cycle in the window directly against every other. It's the top performer on 3 of the " +
      "4 subsets in this comparison.",
    limitations:
      "Generally the most data-hungry of the three sequence architectures. On single-" +
      "condition subsets with fewer distinct degradation patterns to learn from, that hunger " +
      "is less of a handicap than it would be on a much smaller dataset.",
  },
  survival: {
    name: "Survival (Weibull AFT)",
    tagline: "Predicts a full time-to-failure distribution, not a single point estimate",
    grounding:
      "Frames RUL the way reliability engineering actually frames it — train engines are " +
      "observed failures, test engines are right-censored. Its concordance index of roughly " +
      "0.80-0.82 across subsets means it reliably ranks which of two engines fails first, a " +
      "capability none of the point-estimate regressors give you directly.",
    limitations:
      "Never wins on NASA score. A handful of engines get a badly overestimated median " +
      "remaining life, and the NASA score's asymmetric penalty (late predictions cost far " +
      "more than early ones) punishes those outliers heavily. It's kept in this comparison " +
      "deliberately, not because it's the best point predictor here, but because it's the " +
      "only model producing a genuine probability distribution over failure time.",
  },
};

export const NASA_SCORE_EXPLAINER =
  "Every model is scored on the same held-out test set against the true, uncapped RUL from " +
  "RUL_*.txt — not the RUL-capped label models are trained against (see the project README " +
  "for why that distinction is load-bearing: an earlier version of this comparison scored " +
  "against the capped label and understated error on long-lived engines). Ranking is by NASA " +
  "PHM08 score, which penalizes a late prediction (telling a planner an engine has more life " +
  "than it does) far more heavily than an early one, since that's the direction that actually " +
  "risks an in-service failure.";

export const KNOWN_LIMITATIONS = [
  {
    title: "FD002 and FD004 are genuinely harder",
    body:
      "Six operating conditions and, on FD003/FD004, two fault modes — not just an artifact " +
      "of preprocessing. Every model's error roughly doubles versus FD001/FD003, and the " +
      "sequence models' edge over XGBoost is concentrated on the single-condition subsets.",
  },
  {
    title: "Uncertainty intervals under-cover on the harder subsets",
    body:
      "About 76-80% empirical coverage against a 90% target on FD002/FD004, versus 89-96% on " +
      "FD001/FD003. Calibration is filtered to exclude RUL-cap-contaminated rows, but " +
      "residual-distribution shift between the calibration split and the true test set is " +
      "larger on the multi-condition subsets — treat interval widths there as optimistic.",
  },
  {
    title: "The champion changes per subset, on purpose",
    body:
      "There is no single best architecture across all four subsets. That's exactly why this " +
      "project runs a real comparison instead of committing to one model in advance — see " +
      "scripts/compare_models.py.",
  },
];
