# Validation plan — pre-registered predictions (subject: Maxime)

The point of this file: every drug experiment on Maxime below was written down **before** it was run.
Each line states the real-fly result we are comparing against, the mechanism we encoded,
the prediction that follows from the mechanism, and — once run — what actually emerged.
A mismatch is a result, not a bug to tune away. The model is never adjusted to fit a
prediction after the fact; when it fails, the failure is recorded and the proposed model
change is stated as a new hypothesis.

Scope and limits, stated up front:

* The brain is a 139,255-neuron leaky integrate-and-fire network with **random, lateralised
  wiring**. It is FlyWire-*scale*, not the FlyWire connectome. There are no real brain
  regions, no neuromodulators (dopamine, octopamine, serotonin), no ventral nerve cord,
  no peripheral nervous system. Drugs act on network parameters (`brain.params`), never on
  behaviour directly. See `pharmacology.py` for the mechanism table.
* Simulation steps are not milliseconds and dose units are not milligrams. Only the
  **shape** of a result can be compared with the literature: direction, monotonic vs
  biphasic, reversible vs not, ordering of effects. Never absolute numbers.
* This is a simulated insect. Nothing here is medical advice or about any human.

Metrics (all generic, computed by `pharmacology.run_trial`; no experiment-specific code):
`activity` (distance per step), `tortuosity` (path / displacement), `bananas` (found per
trial), `firing_all` and `firing[region]` (fraction of neurons spiking per step),
`synchrony` (fraction of steps on which more than 15 % of all neurons spike together).

---

## P1 — TTX silences the brain and the body

* **Literature:** tetrodotoxin blocks voltage-gated Na⁺ channels; neurons cannot fire.
* **Mechanism encoded:** a dose-dependent fraction of neurons gets `ttx_frac` (cannot spike).
* **Prediction:** firing → ~0, the fly stops (only the body's residual `forward_base` remains).
* **Result (check_drugs.py):** firing 4.91 % → 0.18 % (−96 %); activity 0.089 → 0.045. **Matches.**

## P2 — Tetanus toxin: brain lit, body frozen

* **Literature:** tetanus toxin light chain cleaves synaptobrevin and blocks transmitter
  release; the standard genetic tool for silencing neuronal *output* in *Drosophila*
  (Sweeney et al. 1995, *Neuron*).
* **Mechanism encoded:** `motor_out` → 0 (motor neurons fire but nothing reaches the legs).
* **Prediction:** firing unchanged, movement → 0. The mirror image of P1.
* **Result:** firing 4.91 % → 4.72 %; activity 0.089 → 0.003. **Matches.**

## P3 — Anaesthesia is dose-dependent and reversible

* **Literature:** volatile anaesthetics (isoflurane) produce dose-dependent loss of
  responsiveness in flies that reverses on washout; recovery time grows with dose.
* **Mechanism encoded:** GABA-A potentiation (`g_inh` ↑) and reduced background drive
  (`I_bg` ↓), half-life 300 steps (raised from 120 on 2026-10-28, before the agent's full sweep,
  so that a 400-step trial samples a meaningful exposure; the mechanism is unchanged).
* **Protocol note (2026-10-28):** the first live Muse run showed doses decaying while the model
  was thinking (wall-clock latency leaking into simulation time). The lab now freezes the
  simulation between actions; concentrations decay only during trials. Results before this
  fix are not comparable and are not reported.
* **Prediction:** firing and activity fall under the drug and recover as concentration decays;
  higher dose → longer time to recovery (dose sweep to be run by the agent).
* **Result (single dose):** firing 4.91 % → 1.51 % under drug, 4.65 % after ~7 half-lives;
  activity 0.089 → 0.050 → 0.087. **Reversibility matches.**
* **Preliminary dose sweep (local planner via the lab protocol, 150-step trials, 2026-10-28):**
  activity 0.088 (baseline) → 0.074 (0.25) → 0.056 (0.5) → 0.050 (1.0) → 0.087 (washout);
  firing 4.84 % → 3.64 % → 2.33 % → 1.88 % → 4.66 %. **Dose-dependent and reversible: matches.**
  Dose–recovery-time curve: pending (agent).

## P4 — Picrotoxin: removing inhibition causes seizure-like bursting

* **Literature:** picrotoxin (GABA-A antagonist) is a standard convulsant in *Drosophila*
  seizure models; the signature is synchronous population bursting.
* **Mechanism encoded:** `g_inh` → ~0.1 of baseline at dose 1.0.
* **Prediction:** firing ↑ and `synchrony` ↑ sharply; navigation collapses.
* **Result:** firing 4.91 % → 5.99 % (+22 %), synchrony 0.00 → 0.00, bananas unchanged.
  **Does NOT match.**
* **Why (diagnosis, 2026-10-28):** the network is *drive-dominated*: its activity level is
  set by background current and the refractory period, not held down by inhibition, so
  removing inhibition releases almost nothing. Real cortex-like networks are
  *inhibition-stabilised* — strong recurrent excitation balanced by strong inhibition —
  and seizures need, in addition, a slow variable (spike-frequency adaptation) to carve
  activity into burst–silence cycles.
* **Model change attempted:** balanced recurrence (`REC_K=40, REC_W≈1–6, REC_INH_SCALE=4`)
  plus adaptation (`adapt_inc≈1–3 mV, adapt_tau=60`). Under picrotoxin the population rate
  became bursty (coefficient of variation 0.04 → 1.2–1.9), **but** the balanced central pool
  clamped itself to ~1.5 % firing and the smell signal no longer reached the motor pools:
  steering fell from +0.97 to ≈0 and `check_fly.py` failed. A single-population model cannot
  presently be both a sensitive feed-forward steering pathway and an inhibition-stabilised
  recurrent pool.
* **Status: open.** Adaptation is kept in the code (`adapt_inc`, default 0) so the experiment
  can be repeated. The proposed next model is two populations — a feed-forward
  sensorimotor pathway and a separate balanced recurrent pool that modulates it — which is
  also closer to real anatomy. Not attempted within the hackathon window.

## P5 — Caffeine increases locomotor activity

* **Literature:** caffeine increases waking activity and reduces sleep in flies
  (Shaw et al. 2000, *Science*; Hendricks et al. 2000, *Neuron*).
* **Mechanism encoded:** adenosine-receptor antagonism modelled as background drive `I_bg` ↑.
* **Prediction:** firing ↑, activity ↑, monotonic with dose. (Weak test: nearly follows from
  the mechanism. Included for the dose–response curve, not as strong evidence.)
* **Preliminary (local planner, 150-step trials, 2026-10-28):** activity 0.088 → 0.100 (0.25)
  → 0.107 (0.5) → 0.117 (1.0) → 0.091 (washout); firing 4.84 % → 5.89 % → 6.23 % → 7.47 %.
  **Monotonic increase: matches.** Full 400-step sweep: pending (agent).

## P6 — Ethanol: biphasic locomotion (the one expected to fail)

* **Literature:** low doses of ethanol *increase* locomotion, high doses sedate — a robust
  biphasic curve in *Drosophila* (Wolf, Rodan, Tsai & Heberlein 2002, *J Neurosci*;
  Parr et al. 2001). The low-dose stimulation depends on dopamine and octopamine signalling.
* **Mechanism encoded (deliberately single-class):** GABA-A potentiation (`g_inh` ↑),
  depression of excitatory transmission (`g_exc` ↓), added membrane noise.
* **Prediction from the mechanism:** monotonic decrease in activity with dose — i.e. the model
  should reproduce sedation but **not** the low-dose hyperactivity, because it has no
  neuromodulatory system.
* **What a match/mismatch means:** if the curve is monotonic, the result supports the
  literature's claim that the stimulant phase requires a second mechanism (modulatory
  arousal gain). The proposed model change is one global excitability gain standing in for
  octopamine/dopamine tone, raised at low ethanol concentration. To be added only *after*
  the monotonic result is recorded, and labelled as a second hypothesis.
* **Preliminary (local planner, 150-step trials, 2026-10-28):** activity 0.088 → 0.082 (0.25)
  → 0.079 (0.5) → 0.070 (1.0) → 0.084 (washout). **Monotonic decrease, no low-dose
  stimulation — as predicted, the single-mechanism model DOES NOT reproduce the biphasic
  curve.** This is the expected mismatch; it motivates the second hypothesis above.
  Full 400-step sweep with finer low doses (0.05–0.25): pending (agent).

---

Reference list to verify before submission (do not cite from memory in the write-up):
Sweeney et al. 1995 *Neuron* (tetanus toxin light chain in *Drosophila*); Shaw et al. 2000
*Science* and Hendricks et al. 2000 *Neuron* (fly sleep, caffeine); Wolf et al. 2002
*J Neurosci* (ethanol biphasic locomotion); Parr et al. 2001 (ethanol locomotor assay);
Dorkenwald et al. 2024 *Nature* (FlyWire connectome, for the neuron count only).
