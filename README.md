# Metaverse Neuro Lab — Maxime, a fly built from the FlyWire connectome, in a 3D lab you can run experiments in

**Maxime** is a simulated fruit fly: the real **FlyWire v783 connectome** (139,255 neurons,
3.7 M connections, 50.7 M synapses, signed by each neuron's predicted transmitter) running as a
spiking network, with a body, an arena with a banana she can smell, a pharmacology layer that
perturbs the network by *mechanism*, and a 3D lab (browser or WebXR). In the lab you — or
**Muse Spark**, the lab's experimenter — write an experiment card (hypothesis, conditions,
what to measure, prediction), the lab runs it with repeats while you watch Maxime, and the
notebook gets a page with the measurements, a plain-language analysis and a verdict.

> Honesty first. Drugs change network parameters, never behaviour; behaviour emerges or it
> doesn't. Every mechanism in the substance library carries a confidence label and its
> sources. Negative results are recorded, not tuned away (`validation.md`). The subject is a
> simulated insect: nothing here is medical advice or about any human. Every claim in this
> README is backed by a referee script you can run.

## Run it

```bash
pip3 install numpy scipy pandas matplotlib websockets openai
# FlyWire data (local only, not in the repo): put the v783 CSVs from codex.flywire.ai in ./flywire/
#   neurons, classification, consolidated_cell_types, connections_princeton, coordinates (gz is fine)
# the first run builds flywire/maxime_v783.npz (16 MB) and reuses it afterwards

python3 server.py                 # Maxime (FlyWire wiring) as a lab server: ws://localhost:8765 + http://localhost:8000
python3 server.py --brain random  # the fast fallback fly (random lateralised wiring) - the lab says which one is loaded
# open  http://localhost:8000/lab.html      3D lab; "ENTER VR" on a headset

export META_API_KEY='LLM_...'     # Meta Model API key - or put the line  META_API_KEY=LLM_...  in a file named .env in this folder (git-ignored)
python3 agent.py --chat           # Muse Spark as experimenter: type in the lab's chat box, approve its cards, read its verdicts
python3 agent.py --chat --auto-approve   # Muse runs its own cards without waiting for you
python3 agent.py                  # one-shot: Muse runs the pre-registered dose series and writes reports/<ts>_experiment.md
```

Referees (each prints ALL PASS or NOT YET):

```bash
python3 check_fly.py                  # the fly: firing, steering, lesion, closed loop, free run   (BRAIN=flywire: steering FAILS - see P7)
python3 check_drugs.py                # pharmacology + substance library honesty and mechanisms   (BRAIN=flywire adds the dopamine test)
python3 check_server.py               # the protocol
python3 check_experiments.py          # an experiment card end to end: propose -> run -> verdict -> notebook -> rerun
python3 stage_map.py                  # where the left/right smell signal dies in the real wiring (validation.md P7)
```

## Using the lab

1. **Notebook & experiments** (top-left button). *Experiments* lists every experiment Maxime
   has been through, with its verdict. *New experiment* is a form: title, hypothesis, a
   substance and amount, what to measure, your prediction, repeats, run length → **Write the
   card** → the lab writes the full card (steps, variables, constraints) → **▶ Run it**.
2. While it runs, the wall monitor shows the condition and run, a progress bar and running
   averages; Maxime is reset before every run. When it is done you get the measurements table,
   the analysis, a verdict badge and **↻ Run it again** (re-runs are linked to the original).
   Every experiment is saved as `notebook/<id>.json` + `.md`; `notebook/notebook.md` is the index.
3. **Talk to the experimenter** (the chat box) with `agent.py --chat` running:
   *"does caffeine 0.6 change how much Maxime walks?"* → Muse writes the card, it appears in the
   drawer, you press Run, Muse explains the verdict with the numbers.
   *"look up Adderall"* → Muse asks the lab what it knows (mechanism on Maxime's handles,
   confidence, papers) and, when online, pulls PubChem and Wikipedia text. For a product, Muse
   asks you for the active ingredients on the label — it never guesses what is in a bottle.
4. The table: click a vial to dose, the buttons to wash out / scramble / restore / move the
   banana; the hologram is Maxime's brain with the real neuron positions, lit by live spikes.

**The verdict rule** is one plain sentence, the same for every experiment: a treatment is
SUPPORTED when at least 80 % of its runs fall on the predicted side of the control average
*and* the gap is at least as large as the run-to-run wobble; otherwise NOT SUPPORTED
(direction held but gap too small, or wrong direction) or CAN'T TELL YET (too few repeats).

## What is where

| file | what it is |
|---|---|
| `flywire_brain.py` | builds Maxime from the FlyWire v783 CSVs (regions, signed synapses, transmitter classes, positions) and runs her as a LIF network |
| `fly.py` | the LIF machinery (`FlyBrain`, `params`, `scramble()`, `restore()`), `FlyBody`, `World`; also the random fallback fly |
| `pharmacology.py` | the classic drug table (mechanism only, Hill dose–effect, half-lives), `Pharmacology`, `TrialMeter`, `run_trial` |
| `substances.py` | the substance library (repellents, insecticide, transporter drugs, foods) with confidence labels + PubChem/Wikipedia lookups |
| `experiments.py` | experiment cards: validation, written steps, the verdict rule, the notebook |
| `server.py` | the brain as a server: protocol, experiment engine, notebook, lookups |
| `agent.py` | Muse Spark as experimenter (one-shot or chat), with the `propose_experiment` and `lookup_substance` verbs |
| `lab.html` + `vendor/three` | the 3D lab (Three.js, no build step, WebXR), notebook drawer, chat |
| `protocol.md` | the WebSocket contract every client speaks |
| `validation.md` | pre-registered predictions P1–P7 with results, including the negative ones |
| `check_*.py`, `stage_map.py` | the referees |
| `notebook/`, `reports/` | what Maxime has been through; Muse's one-shot reports |
| `run.py`, `narrator.py`, `fake_server.py` | 2D viewer; optional Muse narration; a scripted stand-in server for UI work |

## How Meta's models are used

* **Muse Spark as the experimenter** (`agent.py`): it gets the lab's verbs and the live numbers,
  looks substances up, writes experiment cards, waits for a human to approve them (or not),
  reads the record and explains the verdict. It cannot touch the fly's position, heading or
  spikes — the protocol has no such verbs — so it can only perturb and measure, like a real
  experimenter.
* **Muse Spark as narrator** (`narrator.py`): one sentence of what the fly senses, from numbers.

## Results so far (details and numbers in `validation.md`)

* TTX silences brain and body; tetanus toxin silences the body while the brain keeps firing;
  isoflurane is dose-dependent and reversible; caffeine raises activity; picrotoxin raises
  firing 4× on the FlyWire brain. Ethanol with a GABA-only mechanism is monotonic (the biphasic
  curve of real flies needs the dopamine/octopamine handles that now exist — to be tested).
* **P7, negative:** in the real wiring the left/right smell difference is lost at the first
  synapse (the banana-glomerulus projection neurons are saturated and get input from both
  antennae), so the FlyWire Maxime does not walk to the banana. Diagnosed stage by stage
  (`stage_map.py`), with the next two hypotheses written down before trying them.
* DEET cuts receptor-neuron firing by ~75 % at full dose; amphetamine moves network firing
  only on the brain that has dopamine neurons.

## Built during the hackathon (Oct 27–Nov 4, 2026)

Everything in this folder. FlyWire data © the FlyWire consortium (Dorkenwald et al. 2024;
Schlegel et al. 2024), used under its terms; the CSVs are not redistributed here. Tools used
to write the code are listed in the submission; Meta's models run *inside* the product as
described above.
