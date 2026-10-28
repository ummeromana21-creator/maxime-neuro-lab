# Validation plan — pre-registered predictions (subject: Maxime)

The point of this file: every drug experiment on Maxime below was written down **before** it was run.
Each line states the real-fly result we are comparing against, the mechanism we encoded,
the prediction that follows from the mechanism, and — once run — what actually emerged.
A mismatch is a result, not a bug to tune away. The model is never adjusted to fit a
prediction after the fact; when it fails, the failure is recorded and the proposed model
change is stated as a new hypothesis.

Scope and limits, stated up front:

* Maxime has two possible brains, and every result says which one it used.
  **FlyWire** (`server.py`, the default): the real FlyWire v783 connectome (Dorkenwald et al.
  2024, *Nature*; Schlegel et al. 2024, *Nature*): 139,255 neurons, 3.73 M connections,
  50.7 M synapses, signed by each neuron's predicted transmitter (Eckstein et al. 2024) with
  the Shiu et al. 2024 (*Nature*) convention (ACh/DA/5-HT/OA excitatory, GABA/Glu inhibitory),
  with leaky integrate-and-fire dynamics. Smell enters through the banana-glomerulus ORNs,
  the legs listen to the descending neurons; there is no ventral nerve cord (the body
  integrates the descending command directly).
  **Random** (`--brain random`): same neuron count, random lateralised wiring, no real
  regions and no transmitter identities. Kept as the fast fallback and as a control.
  In both, drugs act on network parameters (`brain.params`), never on behaviour directly
  (`pharmacology.py`, `substances.py`).
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

## P7 — Does the real wiring steer Maxime toward the banana? (negative result, recorded)

* **Question:** with the FlyWire connectome, ORN → ... → descending neurons, does a left/right
  smell difference at the antennae reach the legs?
* **Prediction:** it should at least reach the lateral horn with a measurable left/right bias
  (Gaudry et al. 2013, *Nature*: ipsilateral ORN release is ~30-40 % stronger, giving PNs a
  bilateral comparison), even if a full walk-to-the-banana needs wind and the central complex.
* **Result (`python3 stage_map.py`, 300 steps per side, seed 1, 2026-10-28):** lateralisation
  index LI = (rate on the banana side − far side) / (sum), averaged over banana-left and
  banana-right:

  | stage | neurons L/R | LI |
  |---|---|---|
  | banana ORNs (DM1/DM2/DM3/DM4/VM2/VA2) | 169/158 | **+0.58** |
  | PNs of those glomeruli | 10/10 | +0.04 |
  | all uniglomerular olfactory PNs | 129/130 | +0.01 |
  | lateral horn (LH*) | 620/607 | +0.01 |
  | mushroom-body Kenyon cells | 2580/2597 | +0.01 |
  | MBONs | 48/48 | +0.01 |
  | LAL | 280/275 | +0.01 |
  | central complex (EPG/PEN/PFL/FC/Δ/ER) | 660/655 | −0.01 |
  | descending neurons | 647/650 | −0.01 |

  **The signal dies at the first synapse.** The banana-glomerulus PNs fire at ~22 % on *both*
  sides whatever side the banana is on: they are saturated, and they receive ORN input from
  both antennae (bilateral ORN projection; the 1.3 : 0.7 ipsilateral bias is not enough once
  the PN is at ceiling). Nothing downstream can recover a difference the PNs no longer carry.
  `check_fly.py` with `BRAIN=flywire` therefore **fails its steering test**, and the lab says so
  in its hello (`wiring: FlyWire v783`). Earlier fixes that were tried and did not change this:
  input-budget normalisation of every neuron's synapses (kept, 300 synapses), compressive
  rather than divisive scaling, the ipsilateral ORN bias, and choosing steering neurons among
  the descending neurons by their side preference (no held-out generalisation).
* **What it means / next hypotheses (not yet done, stated here first):** (1) PN gain control:
  real PNs adapt strongly and are normalised by GABAergic antennal-lobe local neurons
  (Olsen & Wilson 2008, *Nature*); in the model the LNs exist but are as saturated as the PNs.
  A PN/LN-specific refractory period or adaptation is the first thing to test.
  (2) Odour navigation in walking flies is wind-guided (Álvarez-Salvado et al. 2018, *eLife*):
  antennal wind sensing through the Johnston's organ and the central complex, which the
  arena does not provide. A wind field plus JO input is the proper experiment.
  Until one of these is done, every FlyWire result in the notebook is about the *brain's*
  response (firing by region, transmitter class, drug effects — all of which work and pass
  `check_drugs.py`), not about finding bananas.

## The substance library (`substances.py`)

Beyond the seven classic drugs, the lab knows repellents (DEET, picaridin, citronellal, PMD),
a pyrethroid insecticide (permethrin and its relatives), transporter drugs (amphetamine,
bupropion, cocaine) and two things it cannot dose (sugar, salt — taste is not modelled, and
the library says so instead of pretending). Each entry states its mechanism on Maxime's
handles, a confidence label (*established in Drosophila* / *insect evidence* / *inferred* /
*not testable yet*), the papers it rests on, and what is *not* modelled. Two new handles
made this possible: per-transmitter release gains (`g_DA`, `g_OCT`, `g_ACH`, …, using FlyWire's
transmitter prediction for each neuron — 584 dopaminergic, 72 octopaminergic neurons) and
an odour-independent drive into the smell receptor neurons (`I_sens_bg`, for chemicals that
excite ORNs regardless of odour). `check_drugs.py` TEST 8–10 verify that every entry changes
exactly the handles it declares, that washout undoes it, that DEET cuts receptor-neuron
firing, and that amphetamine moves network firing on the FlyWire brain (and does nothing on
the random brain, which has no dopamine neurons — the referee says so).

Pre-registered expectations for the library, to be run as notebook experiments:

* **DEET** should reduce bananas found and receptor firing (`firing.sens_*`), with the brain's
  overall firing nearly unchanged. Real flies: DEET makes attractive odours less attractive
  (Ditzen et al. 2008).
* **Citronellal / PMD** should *raise* receptor-neuron firing while bananas found falls: the
  banana is drowned out rather than hidden. (The aversion itself is not modelled.)
* **Permethrin** should raise firing and activity at low dose and silence the fly at high
  dose (depolarising block) — the classic pyrethroid knock-down shape.
* **Amphetamine / cocaine / bupropion** change firing through dopamine neurons only on the
  FlyWire brain. Direction of the *behavioural* change is not predicted here: with the
  steering problem of P7 unsolved, only firing-based measures count for these.
* **Sugar, salt:** the lab must refuse to dose them and explain why.

---

Reference list to verify before submission (do not cite from memory in the write-up):
Sweeney et al. 1995 *Neuron* (tetanus toxin light chain in *Drosophila*); Shaw et al. 2000
*Science* and Hendricks et al. 2000 *Neuron* (fly sleep, caffeine); Wolf et al. 2002
*J Neurosci* (ethanol biphasic locomotion); Parr et al. 2001 (ethanol locomotor assay);
Dorkenwald et al. 2024 *Nature* and Schlegel et al. 2024 *Nature* (FlyWire connectome and
annotations); Eckstein et al. 2024 *Cell* (transmitter prediction); Shiu et al. 2024 *Nature*
(leaky integrate-and-fire model of the whole connectome); Gaudry et al. 2013 *Nature*
(bilateral olfactory comparison); Olsen & Wilson 2008 *Nature* (antennal-lobe gain control);
Álvarez-Salvado et al. 2018 *eLife* (wind-guided odour navigation); and the citations listed
per entry in `substances.py`.
