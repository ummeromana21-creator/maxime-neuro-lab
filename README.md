# Metaverse Neuro Lab — Maxime, a 139,255-neuron fly you can dose

**Maxime** is a spiking-network fruit fly (FlyWire-scale: 139,255 leaky integrate-and-fire
neurons, ~6.5 M synapses, random **lateralised** wiring) with a body, an arena with a banana, a pharmacology layer
that perturbs the network by mechanism, a 3D lab you can stand in (browser or WebXR), and a
**Muse Spark experimenter** that designs and runs dose–response experiments through the same
command interface a human uses — then writes the lab report.

> Honesty first. The wiring is *not* the FlyWire connectome (it borrows the neuron count and a
> left/right structure). Drugs are modelled as changes to network parameters, never to
> behaviour; behaviour emerges or it doesn't. The subject is a simulated insect: nothing here
> is medical advice or about any human. Every claim in this README is backed by a referee
> script you can run.

## Run it

```bash
pip3 install numpy scipy matplotlib websockets openai
python3 check_fly.py        # 5 tests on the fly: firing, steering, lesion, closed loop, free run -> ALL PASS
python3 check_drugs.py      # 7 tests on the pharmacology                                        -> ALL PASS
python3 run.py              # 2D viewer (keys: S scramble, R restore, B banana, Q quit)

python3 server.py           # the brain as a lab server: ws://localhost:8765 + http://localhost:8000
# then open  http://localhost:8000/lab.html   (3D lab; "ENTER VR" on a headset)

export META_API_KEY='LLM_...'   # Meta Model API
python3 agent.py            # Muse Spark runs the pre-registered experiments and writes reports/<ts>_experiment.md
python3 agent.py --goal "Does nicotine's effect on activity reverse at high dose?"
python3 agent.py --chat     # talk to Muse: type in the lab page's chat box (or this terminal) and watch it work
```

**Chat mode** is the intended way to use the lab: with `server.py`, `agent.py --chat` and the lab page
open, type *"run a dose-response of isoflurane and tell me if it's reversible"* in the box under
*Talk to the experimenter*. Muse designs the series, runs it through the lab's verbs while you watch
Maxime, and answers on the wall with the measured numbers. Reports it decides to write land in
`reports/`.

No key? `server.py` narrates locally and `agent.py` falls back to a fixed dose sweep (the
report says so). `fake_server.py` serves the lab with a scripted fly for UI work without a brain.

## What is where

| file | what it is |
|---|---|
| `fly.py` | `FlyBrain` (LIF network, `params` dict, `scramble()`, `restore()`), `FlyBody` (walls), `World` (smell gradient) |
| `check_fly.py` | the fly's referee: 5 tests, prints ALL PASS or NOT YET |
| `pharmacology.py` | mechanism-only drug table (Hill dose–effect, half-lives), `Pharmacology`, `TrialMeter`, `run_trial` |
| `check_drugs.py` | the drugs' referee: honesty rules + mechanisms |
| `validation.md` | **pre-registered predictions** against the fly literature, with results as they come in |
| `protocol.md` | the WebSocket contract every client speaks |
| `server.py` / `fake_server.py` | the brain as a server / a scripted stand-in |
| `agent.py` | Muse Spark as experimenter: restore → dose → run_trial → … → report |
| `lab.html` + `vendor/three` | the 3D lab (Three.js, no build step, WebXR) |
| `run.py` / `narrator.py` | 2D matplotlib viewer / optional Muse narration of what the fly senses |
| `reports/` | experiment reports written by the agent |

## How Meta's models are used

* **Muse Spark as the experimenter** (`agent.py`): it receives the lab's verbs and the live
  numbers, designs the dose series, runs it, reads the metrics, and writes the report with a
  MATCHED / DID NOT MATCH verdict for each pre-registered prediction. It cannot touch the
  fly's position, heading or spikes — the protocol has no such verbs — so it can only perturb
  and measure, like a real experimenter.
* **Muse Spark as narrator** (`narrator.py`): one sentence of what the fly senses, from numbers.

## Results so far (see `validation.md`)

* TTX silences brain and body; tetanus toxin silences the body while the brain keeps firing;
  isoflurane is dose-dependent and reversible; caffeine raises activity monotonically.
* Ethanol with a GABA-only mechanism gives a monotonic decrease — it does **not** reproduce
  the biphasic (low-dose stimulant) curve real flies show. That was the pre-registered
  expectation: the stimulant phase needs a neuromodulatory mechanism the model lacks.
* Picrotoxin raises firing by ~22 % but produces no seizure-like bursting. Diagnosis: the
  network is drive-dominated, not inhibition-stabilised; a balanced recurrent regime with
  adaptation does burst but clamps the steering pathway. Recorded as an open problem.

## Built during the hackathon (Oct 27–Nov 4, 2026)

Everything in this folder. Tools used to write the code are listed in the submission;
Meta's models run *inside* the product as described above.
# maxime-neuro-lab
