"""
agent.py  -  Muse Spark as the experimenter in the fly lab.

    python3 server.py                                   # in one terminal
    export META_API_KEY='LLM_...'                       # Meta Model API key
    python3 agent.py --goal "Dose-response of isoflurane: firing, activity, bananas; is it reversible?"

The agent connects to the lab like any other client (protocol.md), reads the available drugs
and parameters, and then DESIGNS and RUNS an experiment by issuing the lab's generic verbs:
restore, dose, washout, set_param, scramble, run_trial.  It reads the numbers that come back,
decides the next step, and ends by writing a markdown report to reports/.

Every thought it has is broadcast as a `log` message, so the 3D lab's wall screen shows the
experimenter thinking while the fly runs.

Honesty: the agent cannot touch the fly's position, heading or spikes - the protocol has no
such verbs.  It can only perturb parameters and measure.  Without META_API_KEY it falls back
to a fixed local planner (a plain dose sweep) so the lab still demos; the report says which.
"""
import argparse, asyncio, json, os, re, sys, time
import websockets

DEFAULT_GOAL = ("Run the pre-registered validation experiments P3 (isoflurane: dose-dependent, reversible), "
                "P5 (caffeine: activity rises with dose) and P6 (ethanol: is the activity curve monotonic or biphasic?). "
                "For each drug: restore, baseline trial, then at least three doses, then washout trial. "
                "Report a table of every trial and say for each prediction whether it matched.")

SYSTEM = """You are the experimenter in a virtual neuroscience lab.
The subject is Maxime, a SIMULATED fruit fly: a 139,255-neuron leaky integrate-and-fire network (FlyWire-scale,
random lateralised wiring - NOT the real connectome), a body, and an arena with one banana she can smell.
Refer to her as Maxime.
Nothing here concerns humans or medicine. Use only fly / network vocabulary. Doses are in model units.

You run experiments by replying with EXACTLY ONE JSON object per turn, nothing else:
  {"thought": "<one sentence>", "action": {"cmd": "restore"}}                         original synapses, default params, no drugs
  {"thought": "...", "action": {"cmd": "dose", "drug": "<name>", "amount": <0..2>}}   add drug (concentrations decay over time)
  {"thought": "...", "action": {"cmd": "washout"}}                                   remove all drugs
  {"thought": "...", "action": {"cmd": "set_param", "key": "<param>", "value": <n>}}  lesion-style change of one brain parameter
  {"thought": "...", "action": {"cmd": "scramble"}}                                  send every synapse to a random target
  {"thought": "...", "action": {"cmd": "run_trial", "steps": 400, "label": "<short label>"}}   measure; returns metrics
  {"thought": "...", "action": {"cmd": "report", "markdown": "<full report>"}}       ends the experiment
  {"thought": "<what you want to tell the human>", "action": {"cmd": "say"}}         reply to the human and wait (chat mode only)

run_trial returns: bananas (found during the trial), activity (distance per step), tortuosity (path/displacement),
mean_abs_turn, firing_all (fraction of all neurons spiking per step), firing per region (sens_L, sens_R, central,
motor_L, motor_R), firing_peak, synchrony (fraction of steps on which >15% of neurons spike together), drugs on board.

Time: the simulation is FROZEN while you think and only advances during run_trial, so drug concentrations decay
only during trials (each drug has a half-life in steps; the result reports the concentration left at the end).
Method rules: start with restore and a baseline run_trial; restore before every new condition; dose right before the
trial; at least three doses for a dose-response; 400 steps per trial unless you state a reason;
compare each result with baseline in your thought. Finish with a report containing a markdown table of EVERY trial
(label, drugs, activity, bananas, firing_all, synchrony) and, for each prediction you were testing, one line saying
MATCHED or DID NOT MATCH and why. Keep thoughts to one sentence. Never invent numbers: only use the ones returned.
"""


class Lab:
    """Thin client over protocol.md."""
    def __init__(self, url):
        self.url, self.ws, self.hello, self.n_cmd = url, None, None, 0
        self.pending = {}                       # (type, id) -> Future
        self.human = asyncio.Queue()            # messages typed by a human in the lab or the terminal

    async def connect(self):
        self.ws = await websockets.connect(self.url, max_size=2 ** 20)
        self.hello = json.loads(await self.ws.recv())
        assert self.hello.get('type') == 'hello', self.hello
        asyncio.create_task(self._reader())
        return self.hello

    async def _reader(self):
        """One reader for the whole connection: routes acks/results to waiting commands,
        human chat lines to the queue, ignores the 30 Hz state stream."""
        try:
            async for raw in self.ws:
                m = json.loads(raw)
                t = m.get('type')
                if t in ('ack', 'result'):
                    fut = self.pending.pop((t, m.get('id')), None)
                    if fut and not fut.done():
                        fut.set_result(m)
                elif t == 'log' and m.get('source') == 'user':
                    await self.human.put(m.get('text', ''))
        except Exception as e:
            for fut in self.pending.values():
                if not fut.done():
                    fut.set_exception(e)
            await self.human.put(None)         # None = connection gone

    async def cmd(self, **kw):
        self.n_cmd += 1
        cid = f"a{self.n_cmd}"
        loop = asyncio.get_running_loop()
        fut_ack = self.pending[('ack', cid)] = loop.create_future()
        fut_res = None
        if kw.get('cmd') == 'run_trial':
            fut_res = self.pending[('result', cid)] = loop.create_future()
        await self.ws.send(json.dumps(dict(id=cid, **kw)))
        ack = await asyncio.wait_for(fut_ack, 30)
        if not ack['ok']:
            self.pending.pop(('result', cid), None)
            return {'error': ack['error']}
        if fut_res is not None:
            res = await asyncio.wait_for(fut_res, 900)
            return res['metrics']
        return ack['data']

    async def log(self, text, source='agent'):
        await self.ws.send(json.dumps({'cmd': 'log', 'source': source, 'text': text}))


def compact(metrics):
    """Metrics as a short dict the model can read without noise."""
    if 'error' in metrics:
        return metrics
    f = metrics.get('firing', {})
    out = {'bananas': metrics['bananas'], 'activity': round(metrics['activity'], 4),
           'tortuosity': None if metrics.get('tortuosity') is None else round(metrics['tortuosity'], 2),
           'mean_abs_turn': round(metrics['mean_abs_turn'], 3),
           'firing_all': round(metrics['firing_all'], 4), 'firing_peak': round(metrics['firing_peak'], 4),
           'firing': {k: round(v, 4) for k, v in f.items()}, 'synchrony': round(metrics['synchrony'], 3),
           'drugs': metrics.get('drugs', {}), 'scrambled': metrics.get('scrambled', False)}
    return out


class MuseExperimenter:
    """Muse Spark decides every step."""
    def __init__(self):
        from openai import OpenAI
        key = os.environ['META_API_KEY']
        base = os.environ.get('META_BASE_URL', 'https://api.meta.ai/v1')
        self.model = os.environ.get('META_MODEL', 'muse-spark-1.3')
        self.client = OpenAI(api_key=key, base_url=base)
        self.messages = []
        self.name = f"Muse ({self.model})"

    def start(self, hello, goal, chat=False):
        drugs = "\n".join(f"  - {k}: {v['mechanism']} (half-life {v['half_life']} steps)" for k, v in hello['drugs'].items())
        params = ", ".join(f"{k}={v}" for k, v in hello['params'].items())
        intro = f"Drugs available:\n{drugs}\n\nBrain parameters (defaults): {params}\n\n"
        if chat:
            self.messages = [{"role": "system", "content": SYSTEM + CHAT_RULES},
                             {"role": "user", "content": intro + "You are in chat mode: wait for the human. Nothing to do until a HUMAN message arrives."},
                             {"role": "assistant", "content": '{"thought": "Ready.", "action": {"cmd": "say"}}'}]
        else:
            self.messages = [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": intro + f"GOAL: {goal}\n\nBegin. Reply with one JSON object."}]

    def on_human(self, text):
        self.messages.append({"role": "user", "content": f"HUMAN: {text}"})

    def _trim(self, keep=60):
        """Keep the system prompt and the intro; drop the oldest of the rest when the chat gets long."""
        if len(self.messages) > keep + 3:
            self.messages = self.messages[:3] + self.messages[-keep:]

    def next(self, feedback=None):
        if feedback is not None:
            self.messages.append({"role": "user", "content": feedback})
        self._trim()
        for attempt in range(3):
            r = self.client.chat.completions.create(model=self.model, messages=self.messages,
                                                    max_tokens=1200, temperature=0.3)
            text = r.choices[0].message.content
            m = re.search(r'\{.*\}', text, re.S)
            try:
                obj = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                obj = None
            if obj and 'action' in obj:
                self.messages.append({"role": "assistant", "content": text})
                return obj
            self.messages.append({"role": "assistant", "content": text})
            self.messages.append({"role": "user", "content": "That was not a single valid JSON object with an 'action'. Reply again with exactly one JSON object."})
        return {"thought": "could not produce a valid action", "action": {"cmd": "report", "markdown": "Aborted: the model did not return valid actions."}}


CHAT_RULES = """
CHAT MODE. A human in the lab talks to you. Messages from them arrive as 'HUMAN: ...'. When they ask for an
experiment, run it with the action verbs, one per turn, reading each RESULT; when it is done, or when they only
ask a question, reply with {"action": {"cmd": "say"}} and put your answer in "thought" (2-4 sentences, plain
language, with the actual numbers). Always say what you are about to do before a long series. Never claim a
result you have not measured. If a request is not something the lab can do, say so.
"""


class MockExperimenter:
    """No API key: understands a few plain phrases so the chat plumbing can be demoed. Not Muse."""
    name = "mock experimenter (no META_API_KEY - not Muse)"
    def __init__(self): self.queue, self.last = [], None
    def start(self, hello, goal): self.drugs = list(hello['drugs'])
    def on_human(self, text):
        t = text.lower(); self.queue = []
        import re as _re
        m = _re.search(r'(sweep|dose[- ]response)\s+(\w+)', t)
        if m and m.group(2) in self.drugs:
            d = m.group(2); self.queue = [{"cmd": "restore"}, {"cmd": "run_trial", "steps": 400, "label": "baseline"}]
            for dose in (0.25, 0.5, 1.0):
                self.queue += [{"cmd": "restore"}, {"cmd": "dose", "drug": d, "amount": dose}, {"cmd": "run_trial", "steps": 400, "label": f"{d} {dose}"}]
            self.queue += [{"cmd": "washout"}, {"cmd": "run_trial", "steps": 400, "label": f"{d} washout"}]
        elif (m := _re.search(r'dose\s+(\w+)\s*([\d.]+)?', t)) and m.group(1) in self.drugs:
            self.queue = [{"cmd": "dose", "drug": m.group(1), "amount": float(m.group(2) or 0.5)}, {"cmd": "run_trial", "steps": 400, "label": f"{m.group(1)} {m.group(2) or 0.5}"}]
        elif 'baseline' in t: self.queue = [{"cmd": "restore"}, {"cmd": "run_trial", "steps": 400, "label": "baseline"}]
        elif 'scramble' in t: self.queue = [{"cmd": "scramble"}, {"cmd": "run_trial", "steps": 400, "label": "scrambled"}]
        elif 'restore' in t: self.queue = [{"cmd": "restore"}]
        elif 'washout' in t or 'wash out' in t: self.queue = [{"cmd": "washout"}]
        else:
            self.queue = [{"cmd": "say", "_text": "I'm the mock experimenter (no META_API_KEY). Try: 'baseline', 'dose caffeine 0.5', 'sweep isoflurane', 'scramble', 'restore', 'washout'."}]
    def next(self, feedback=None):
        if feedback and feedback.startswith("RESULT [") and self.last and self.last.get('cmd') == 'run_trial':
            self.results = getattr(self, 'results', []) + [feedback]
        if self.queue:
            a = self.queue.pop(0); self.last = a
            if a['cmd'] == 'say': return {"thought": a['_text'], "action": {"cmd": "say"}}
            return {"thought": f"(mock) {a['cmd']} {a.get('drug','')} {a.get('amount','')} {a.get('label','')}".strip(), "action": a}
        rs = getattr(self, 'results', []); self.results = []
        summary = " ".join(r.split(': ', 1)[0].replace('RESULT ', '') + ' ' + r.split('"activity": ')[1].split(',')[0] if '"activity": ' in r else '' for r in rs)
        return {"thought": f"Done. Activity per trial: {summary}" if rs else "Done.", "action": {"cmd": "say"}}


class LocalPlanner:
    """No API key: a fixed dose sweep so the lab still demos. Says so in the report."""
    name = "local planner (no META_API_KEY)"

    def __init__(self):
        self.plan, self.i, self.rows = [], 0, []

    def start(self, hello, goal):
        drugs = [d for d in ('isoflurane', 'caffeine', 'ethanol') if d in hello['drugs']]
        self.plan = [("restore", {"cmd": "restore"}), ("baseline", {"cmd": "run_trial", "steps": 400, "label": "baseline"})]
        for d in drugs:
            for dose in (0.25, 0.5, 1.0):
                self.plan += [(f"restore before {d} {dose}", {"cmd": "restore"}),
                              (f"dose {d} {dose}", {"cmd": "dose", "drug": d, "amount": dose}),
                              (f"measure {d} {dose}", {"cmd": "run_trial", "steps": 400, "label": f"{d} {dose}"})]
            self.plan += [("washout", {"cmd": "washout"}),
                          (f"measure after washout of {d}", {"cmd": "run_trial", "steps": 400, "label": f"{d} washout"})]

    def next(self, feedback=None):
        if feedback and feedback.startswith("RESULT"):
            self.rows.append(feedback)
        if self.i < len(self.plan):
            thought, action = self.plan[self.i]; self.i += 1
            return {"thought": thought, "action": action}
        return {"thought": "sweep complete", "action": {"cmd": "report", "markdown":
                "# Local planner report\n\n(No META_API_KEY: this sweep was run by a fixed script, not by Muse.)\n\n"
                + "\n".join(f"- {r}" for r in self.rows)}}


ALLOWED = ('restore', 'dose', 'washout', 'set_param', 'scramble', 'run_trial', 'reset_fly')


async def run_series(lab, brain, a, trials, transcript, feedback=None):
    """Drive the experimenter until it says something, reports, or runs out of actions.
    Returns (kind, text): kind is 'say', 'report' or 'limit'."""
    await lab.cmd(cmd='speed', sps=0)                       # run the brain flat out during trials...
    await lab.cmd(cmd='pause')                              # ...and freeze it while the experimenter thinks
    try:
        for i in range(a.max_actions):
            step = brain.next(feedback)
            thought, action = step.get('thought', ''), step.get('action', {})
            c = action.get('cmd')
            print(f"[{len(transcript)+1}] {thought}\n     -> {json.dumps(action)[:160]}")
            transcript.append((thought, action))
            if c == 'say':
                return 'say', thought
            if c == 'report':
                return 'report', action.get('markdown', thought)
            await lab.log(thought)
            if c not in ALLOWED:
                feedback = f"ERROR: '{c}' is not an allowed cmd. Allowed: {ALLOWED} plus say/report."
                continue
            if c == 'run_trial':
                if a.steps:
                    action['steps'] = a.steps
                action['pause_after'] = True                # the server freezes again the moment the trial ends
            result = await lab.cmd(**action)
            if c == 'run_trial' and 'error' not in result:
                trials.append((action.get('label', ''), result))
                await lab.log(f"result [{action.get('label','')}]: bananas {result['bananas']}, activity {result['activity']:.3f}, "
                              f"firing {100*result['firing_all']:.2f}%, synchrony {result['synchrony']:.2f}", source='lab')
                feedback = f"RESULT [{action.get('label','')}]: {json.dumps(compact(result))}"
            else:
                feedback = f"RESULT: {json.dumps(result)}"
            print(f"     <- {feedback[:200]}")
        return 'limit', "Reached the action limit without finishing."
    finally:
        await lab.cmd(cmd='speed', sps=60); await lab.cmd(cmd='resume')


def write_report(hello, brain, goal, report, trials, transcript):
    os.makedirs('reports', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join('reports', f"{ts}_experiment.md")
    with open(path, 'w') as f:
        f.write(f"# Experiment report — subject: {hello.get('name', 'Maxime')}\n\n*Experimenter:* {brain.name}  \n*Goal:* {goal}  \n*Lab mode:* {hello['mode']}  \n*Date:* {ts}\n\n")
        f.write("## Report by the experimenter\n\n" + report + "\n\n## Trials (as measured by the lab)\n\n")
        f.write("| label | drugs | bananas | activity | tortuosity | firing_all | synchrony | motor_L | motor_R |\n|---|---|---|---|---|---|---|---|---|\n")
        for label, m in trials:
            tort = '-' if m.get('tortuosity') is None else f"{m['tortuosity']:.2f}"
            drugs = ", ".join(f"{k} {v:.2f}" for k, v in m.get('drugs', {}).items()) or "none"
            f.write(f"| {label} | {drugs} | {m['bananas']} | {m['activity']:.3f} | {tort} | {100*m['firing_all']:.2f}% | "
                    f"{m['synchrony']:.2f} | {100*m['firing']['motor_L']:.2f}% | {100*m['firing']['motor_R']:.2f}% |\n")
        f.write("\n## Transcript\n\n" + "\n".join(f"{i+1}. {t} -> `{json.dumps(ac)[:200]}`" for i, (t, ac) in enumerate(transcript)) + "\n")
    return path


async def stdin_lines(lab):
    """Terminal input in chat mode: forwarded to the lab as a human message (so it shows on the wall too)."""
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            await asyncio.sleep(3600); continue
        line = line.strip()
        if line:
            await lab.log(line, source='user')


async def chat(lab, hello, brain, a):
    print("Chat mode. Type here, or in the lab page's chat box. Ctrl-C to stop.\n")
    await lab.log(f"{brain.name} is listening — ask for an experiment, e.g. 'dose-response of caffeine', 'what happens if you scramble her synapses?'")
    brain.start(hello, None, chat=True) if isinstance(brain, MuseExperimenter) else brain.start(hello, None)
    asyncio.create_task(stdin_lines(lab))
    trials, transcript = [], []
    while True:
        text = await lab.human.get()
        if text is None:
            print("lab connection closed"); return
        print(f"\nHUMAN: {text}")
        brain.on_human(text)
        kind, out = await run_series(lab, brain, a, trials, transcript)
        if kind == 'report':
            path = write_report(hello, brain, text, out, trials, transcript)
            summary = out.strip().splitlines()[0][:300] if out.strip() else 'report written'
            await lab.log(summary); await lab.log(f"report written: {path}")
            print(f"MUSE: {summary}\n(report: {path})")
            trials, transcript = [], []
        else:
            await lab.log(out)
            print(f"MUSE: {out}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='ws://localhost:8765')
    ap.add_argument('--goal', default=DEFAULT_GOAL)
    ap.add_argument('--chat', action='store_true', help='talk to the experimenter instead of giving it one goal')
    ap.add_argument('--max-actions', type=int, default=60)
    ap.add_argument('--steps', type=int, default=None, help='override trial length for every run_trial')
    a = ap.parse_args()

    lab = Lab(a.url)
    hello = await lab.connect()
    print(f"Connected to lab ({hello['mode']}): subject {hello.get('name', 'Maxime')}, {hello['n']:,} neurons, drugs {sorted(hello['drugs'])}")
    key = os.environ.get('META_API_KEY', '')
    brain = None
    if key and not key.startswith('your_'):
        try:
            brain = MuseExperimenter()
        except Exception as e:
            print(f"Muse unavailable ({e!r}); falling back")
    if brain is None:
        brain = MockExperimenter() if a.chat else LocalPlanner()
    print(f"Experimenter: {brain.name}")
    await lab.log(f"experimenter online: {brain.name}")

    if a.chat:
        await chat(lab, hello, brain, a)
        return

    print(f"Goal: {a.goal}\n")
    await lab.log(f"goal: {a.goal}")
    brain.start(hello, a.goal)
    trials, transcript = [], []
    kind, out = await run_series(lab, brain, a, trials, transcript)
    report = out if kind == 'report' else f"({kind}) {out}"
    await lab.cmd(cmd='restore')
    path = write_report(hello, brain, a.goal, report, trials, transcript)
    await lab.log(f"report written: {path}")
    print(f"\nReport: {path}")
    await lab.ws.close()


if __name__ == '__main__':
    asyncio.run(main())
