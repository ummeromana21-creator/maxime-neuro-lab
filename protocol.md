# Lab protocol — how everything talks to the fly

One brain, many clients. `server.py` runs the fly (`fly.py` + `pharmacology.py`) and speaks
this protocol. Every client — the 3D lab (`lab.html`), the Muse experimenter (`agent.py`),
a judge with a WebSocket console — uses the **same verbs**. No experiment is written into
the server; experiments are composed from these verbs by whoever is connected.

* WebSocket: `ws://localhost:8765` — JSON text frames.
* HTTP: `http://localhost:8000/` — the server also serves this folder, so open
  `http://localhost:8000/lab.html` for the 3D lab.
* `fake_server.py` speaks the identical protocol with a scripted fly and no brain, so the
  lab can be developed and demoed without the simulation.

Coordinates: the Python arena is `-10..10` in x and y (walls at ±10). The 3D lab maps that
to a 1 m box: `metres = units / 20`. Angles are radians, 0 = +x, counter-clockwise positive.

---

## Server → client

### `hello` — once, on connect

```json
{"type":"hello", "mode":"live", "name":"Maxime",
 "n":139255, "arena":10.0, "catch_radius":0.9,
 "regions":{"sens_L":[0,5000],"sens_R":[5000,10000],"central":[10000,120000],
            "motor_L":[120000,129500],"motor_R":[129500,139255]},
 "sample":{"size":2000,"bands":[["sens_L",200],["sens_R",200],["central",1000],
                                ["motor_L",300],["motor_R",300]]},
 "params":{"I_bg":1.3, "g_exc":0.5, ...},
 "drugs":{"picrotoxin":{"mechanism":"...","half_life":400}, ...}}
```

`sample` describes a fixed sample of neurons used for the brain display: 2000 neurons in
five contiguous bands, in that order. `fired` (below) indexes into this sample.

### `state` — about 30 times per second

```json
{"type":"state", "t":12345, "sps":78.2,
 "fly":{"x":1.2,"y":-3.4,"angle":0.7}, "banana":{"x":4.0,"y":2.5}, "dist":5.1, "caught":7,
 "motor":{"forward":0.58,"turn":-0.21},
 "rates":{"sens_L":0.07,"sens_R":0.05,"central":0.043,"motor_L":0.05,"motor_R":0.04,"all":0.045},
 "fired":[3,17,41, ...],
 "scrambled":false, "drugs":{"ttx":0.42}, "params":{...},
 "thought":"banana on my right, turning", "paused":false,
 "trial":null}
```

* `sps` = brain steps per second actually running.
* `fired` = indices (into the sample) of neurons that spiked in **any** step since the previous
  `state` message (a union), so a 30 Hz display sees every spike.
* `trial` = `null`, or `{"id":"...","progress":0.42,"steps":400}` while a trial runs.

### `ack` — reply to every command

```json
{"type":"ack", "id":"c17", "cmd":"dose", "ok":true, "error":null, "data":{"picrotoxin":0.5}}
```

### `result` — when a trial finishes

```json
{"type":"result", "id":"t3", "label":"picrotoxin 0.5", "metrics":{
   "steps":400, "bananas":3, "activity":0.087, "tortuosity":2.9, "mean_abs_turn":0.31,
   "firing_all":0.047, "firing_peak":0.061,
   "firing":{"sens_L":0.07,"sens_R":0.07,"central":0.043,"motor_L":0.05,"motor_R":0.05},
   "synchrony":0.0, "drugs":{"picrotoxin":0.44}, "scrambled":false}}
```

Metric definitions live in `pharmacology.TrialMeter`; the server, `run_trial()` and
`check_drugs.py` all use it, so numbers are comparable everywhere.

### `log` — a line for the lab's wall screen

```json
{"type":"log", "source":"agent", "text":"Dosing picrotoxin 0.5, running 400 steps", "ts":1730130000.1}
```

Relayed to every client. The agent posts its reasoning this way; the lab displays it.

---

## Client → server

Every command is an object with `"cmd"` and an optional `"id"` (echoed in the `ack`).

| cmd | fields | effect |
|---|---|---|
| `scramble` | — | send every synapse to a random target (control experiment) |
| `restore` | — | original synapses, default params, no drugs, TTX mask cleared |
| `reset_fly` | — | fly to the centre with a random heading, banana somewhere new |
| `move_banana` | `x`, `y` (optional) | place the banana; omit both for a random spot |
| `dose` | `drug`, `amount` | add `amount` (0–2) of a drug from `hello.drugs` |
| `washout` | — | remove all drugs |
| `set_param` | `key`, `value` | change a baseline brain parameter (a lesion-style manipulation) |
| `get_params` | — | ack `data` = current params and drug concentrations |
| `run_trial` | `steps` (default 400), `label` | measure the next `steps` steps; `result` follows |
| `cancel_trial` | — | abandon the running trial (no `result`) |
| `speed` | `sps` | cap brain steps per second (0 = unlimited; default 60) |
| `pause` / `resume` | — | freeze / unfreeze the simulation |
| `log` | `text`, `source` | broadcast a line to all clients |
| `ping` | — | ack |
| `propose_experiment` | `card` | validate a card (below); ack `data` = `{id, card, steps}`; broadcasts `proposal` |
| `run_experiment` | `exp` | run a proposed or past experiment (a past one becomes a re-run with `rerun_of`); live progress in `state.experiment`; `experiment_started`, one `experiment_run` per run, then `experiment_done` with the record and the notebook list |
| `discard_proposal` | `exp` | drop a proposal; broadcasts `proposal_discarded` |
| `cancel_experiment` | — | stop the running experiment; broadcasts `experiment_cancelled` |
| `get_experiment` | `exp` | ack `data` = `{record, markdown, steps}` |
| `list_notebook` | — | ack `data` = `{notebook}` (one summary per experiment) |
| `lookup` | `name`, `offline` (optional) | what the lab knows about a substance (`match`: library entry with mechanism, confidence, evidence, `testable`, `dose_as`) plus `sources` from PubChem and Wikipedia when online; runs off the simulation thread |

An experiment **card**: `title`, `hypothesis`, `description`, `control {name, setup}`,
`treatments [{name, setup}]` (1–6), `measure` (one of `hello.measures`), `prediction`
(`higher` / `lower` / `different`), `repeats` (3–10), `steps` (100–2000), optional `variables`,
`constraints`, `author`. A `setup` is a list of the verbs `restore`, `dose`, `washout`,
`set_param`, `scramble`. The lab resets Maxime (`restore` + `reset_fly`) before every run, applies
the condition's setup, lets her settle 30 steps, then measures `steps` steps. The verdict rule
is in `experiments.py` (`analyse`) and in the README.

Rules the server enforces: a drug or `set_param` only ever changes `brain.params`; nothing
can set the fly's position, heading or spikes directly (there is deliberately no such
command). `run_trial` records whatever state the fly is in — the caller composes the
experiment: `restore` → `dose` → `run_trial`, repeat across doses.

---

## A minimal experiment, by hand

```
> {"cmd":"restore"}
> {"cmd":"run_trial","steps":400,"label":"baseline"}
< {"type":"result", "metrics":{...}}
> {"cmd":"dose","drug":"isoflurane","amount":0.8}
> {"cmd":"run_trial","steps":400,"label":"isoflurane 0.8"}
< {"type":"result", "metrics":{...}}
> {"cmd":"washout"}
> {"cmd":"run_trial","steps":400,"label":"after washout"}
```

That sequence — baseline, drug, washout — was never written into any file in this folder.
