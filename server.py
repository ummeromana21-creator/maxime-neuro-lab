"""
server.py  -  runs the fly and speaks protocol.md to every connected client.

    python3 server.py                # brain on ws://localhost:8765, files on http://localhost:8000
    python3 server.py --sps 0        # run the brain as fast as it can (default cap 60 steps/s)
    python3 server.py --seed 3 --no-narrator

Then open  http://localhost:8000/lab.html  (3D lab)  and/or  python3 agent.py  (experimenter).

The simulation runs in its own thread. Commands from clients are queued and applied
between brain steps; a snapshot of the state is broadcast ~30 times a second.
Nothing in this file decides what the fly does: it only runs fly.py and pharmacology.py
and relays verbs.  pip3 install websockets
"""
import argparse, asyncio, http.server, json, os, queue, sys, threading, time
import numpy as np
import experiments as EX

try:
    import websockets
except ImportError:
    sys.exit("pip3 install websockets   (the lab server needs it)")

WS_PORT, HTTP_PORT, BROADCAST_HZ = 8765, 8000, 30
SAMPLE_BANDS = [('sens_L', 200), ('sens_R', 200), ('central', 1000), ('motor_L', 300), ('motor_R', 300)]


# =========================================================================================
# the simulation thread
# =========================================================================================
class Sim:
    mode = 'live'

    def __init__(self, seed=None, sps_cap=60, narrate=True, brain='flywire'):
        import fly, pharmacology as ph
        from narrator import Narrator
        self.ph = ph
        t0 = time.time()
        if brain == 'flywire':
            import flywire_brain
            print("Building Maxime's brain from the FlyWire v783 connectome...")
            self.brain = flywire_brain.FlyWireBrain(seed=seed)
            print("  " + self.brain.describe())
        else:
            print("Building Maxime's brain (synthetic, random lateralised wiring)...")
            self.brain = fly.FlyBrain(seed=seed)
        self.brain_kind = brain
        self.body = fly.FlyBody()
        self.world = fly.World(seed=seed)
        self.world.randomize_sugar()
        self.body.leg_phases = np.array([0, np.pi, np.pi, 0, 0, np.pi])
        self.pharm = ph.Pharmacology(self.brain)
        self.narrator = Narrator(enabled=narrate)
        print(f"  {self.brain.n:,} neurons, {self.brain.W.nnz:,} synapses, {time.time()-t0:.1f}s")

        # fixed neuron sample for the brain display
        idx, self.bands = [], []
        for name, cnt in SAMPLE_BANDS:
            a, b = self.brain.regions[name]
            cnt = min(cnt, b - a)
            idx.append(np.linspace(a, b - 1, cnt).astype(int)); self.bands.append([name, cnt])
        self.sample_idx = np.concatenate(idx)
        self.fired_union = np.zeros(self.sample_idx.size, dtype=bool)

        self.lock = threading.Lock()
        self.cmd_q, self.out_q = queue.Queue(), queue.Queue()
        self.t, self.caught, self.sps, self.sps_cap = 0, 0, 0.0, sps_cap
        self.paused, self.running = False, True
        self.trial = None
        self.exp = None                       # running experiment (see _exp_* below)
        self.proposals = {}                   # id -> card awaiting approval
        self.notebook = EX.Notebook()
        self.snap = {}
        self._snapshot(np.zeros(self.brain.n, bool), np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0)

    # ---- hello ---------------------------------------------------------------------------
    def hello(self):
        import fly
        meta = getattr(self.brain, 'meta', None)
        sample = {'size': int(self.sample_idx.size), 'bands': self.bands}
        if meta is not None:                                            # real brain: real positions and names
            xyz = meta['xyz'][self.sample_idx]
            sample['positions'] = np.round(xyz, 3).tolist()
            sample['types'] = [str(t) for t in meta['primary_type'][self.sample_idx]]
            sample['sides'] = [str(t) for t in meta['side'][self.sample_idx]]
        return {'type': 'hello', 'mode': self.mode, 'name': fly.NAME, 'n': self.brain.n, 'arena': self.body.ARENA,
                'catch_radius': self.world.catch_radius,
                'wiring': getattr(self.brain, 'wiring', 'unknown'), 'brain': self.brain_kind,
                'subject': self.brain.describe() if hasattr(self.brain, 'describe') else 'synthetic 139,255-neuron network, random lateralised wiring',
                'regions': {k: list(v) for k, v in self.brain.regions.items()},
                'sample': sample,
                'params': {k: float(v) for k, v in self.brain.params.items()},
                'drugs': {k: {'mechanism': d['mechanism'], 'half_life': d['half_life'], 'kind': d.get('kind', 'classic drug'), 'confidence': d.get('confidence', 'established'), 'evidence': d.get('evidence', [])}
                          for k, d in self.ph.DRUGS.items()},
                'measures': EX.MEASURES, 'setup_verbs': list(EX.ALLOWED_SETUP),
                'notebook': self.notebook.list()}

    # ---- commands (run on the sim thread, between steps) ----------------------------------
    def handle(self, cmd):
        c = cmd.get('cmd')
        b, w, body, pharm = self.brain, self.world, self.body, self.pharm
        if c == 'ping':
            return {}
        if c == 'scramble':
            b.scramble(); return {'scrambled': True}
        if c == 'restore':
            b.restore(); pharm.reset(); return {'scrambled': False}
        if c == 'reset_fly':
            body.x, body.y = 0.0, 0.0
            body.angle = float(np.random.uniform(-np.pi, np.pi))
            w.randomize_sugar(); return {'x': 0.0, 'y': 0.0, 'angle': body.angle}
        if c == 'move_banana':
            if 'x' in cmd and 'y' in cmd:
                lim = body.ARENA - 1.0
                w.sugar_x = float(np.clip(cmd['x'], -lim, lim)); w.sugar_y = float(np.clip(cmd['y'], -lim, lim))
            else:
                w.randomize_sugar()
            return {'x': w.sugar_x, 'y': w.sugar_y}
        if c == 'dose':
            conc = pharm.dose(str(cmd['drug']), float(cmd['amount'])); return {cmd['drug']: conc}
        if c == 'washout':
            pharm.washout(); return {'drugs': {}}
        if c == 'set_param':
            pharm.set_param(str(cmd['key']), float(cmd['value'])); return {cmd['key']: b.params[cmd['key']]}
        if c == 'get_params':
            return {'params': {k: float(v) for k, v in b.params.items()}, 'drugs': pharm.state()}
        if c == 'propose_experiment':
            card, err = EX.validate(cmd.get('card'), self.ph.DRUGS, b.params)
            if err: raise ValueError(err)
            eid = self.notebook.new_id(card['title'])
            self.proposals[eid] = card
            msg = {'type': 'proposal', 'id': eid, 'card': card, 'steps': EX.steps_text(card)}
            self.out_q.put(msg)
            return {'id': eid, 'card': card, 'steps': EX.steps_text(card)}
        if c == 'discard_proposal':
            self.proposals.pop(str(cmd.get('exp')), None)
            self.out_q.put({'type': 'proposal_discarded', 'id': cmd.get('exp')}); return {}
        if c == 'run_experiment':
            if self.exp is not None:
                raise RuntimeError('an experiment is already running')
            if self.trial is not None:
                raise RuntimeError('a trial is running; wait for it')
            eid = cmd.get('exp'); rerun_of = None
            if eid and eid in self.proposals:
                card = self.proposals.pop(eid)
            elif eid and self.notebook.load(eid):
                card = self.notebook.load(eid)['card']; rerun_of = eid; eid = self.notebook.new_id(card['title'])
            else:
                card, err = EX.validate(cmd.get('card'), self.ph.DRUGS, b.params)
                if err: raise ValueError(err)
                eid = self.notebook.new_id(card['title'])
            self._exp_start(eid, card, rerun_of, bool(cmd.get('pause_after', False)))
            return {'id': eid, 'card': card}
        if c == 'cancel_experiment':
            if self.exp: self.out_q.put({'type': 'experiment_cancelled', 'id': self.exp['id']})
            self.exp = None; self.trial = None; return {}
        if c == 'list_notebook':
            return {'notebook': self.notebook.list()}
        if c == 'get_experiment':
            rec = self.notebook.load(str(cmd.get('exp')))
            if rec is None: raise KeyError('no such experiment')
            return {'record': rec, 'markdown': self.notebook.markdown(rec), 'steps': EX.steps_text(rec['card'])}
        if c == 'run_trial':
            if self.trial is not None:
                raise RuntimeError('a trial is already running')
            if self.exp is not None:
                raise RuntimeError('an experiment is running; wait for it')
            steps = int(cmd.get('steps', 400))
            if not 10 <= steps <= 20000:
                raise ValueError('steps must be 10..20000')
            self.trial = {'id': cmd.get('id'), 'label': str(cmd.get('label', '')), 'steps': steps,
                          'meter': self.ph.TrialMeter(b, body), 'pause_after': bool(cmd.get('pause_after', False))}
            self.paused = False
            return {'steps': steps}
        if c == 'cancel_trial':
            self.trial = None; return {}
        if c == 'speed':
            self.sps_cap = float(cmd.get('sps', 60)); return {'sps': self.sps_cap}
        if c == 'pause':
            self.paused = True; return {'paused': True}
        if c == 'resume':
            self.paused = False; return {'paused': False}
        raise ValueError(f"unknown cmd '{c}'")

    # ---- experiment engine ----------------------------------------------------------------
    def _exp_start(self, eid, card, rerun_of, pause_after):
        conds = [card['control']] + card['treatments']
        plan = [(cond['name'], rep) for cond in conds for rep in range(card['repeats'])]
        self.exp = {'id': eid, 'card': card, 'conds': {c['name']: c for c in conds}, 'plan': plan, 'i': 0,
                    'phase': 'setup', 'settle': 0, 'meter': None, 'results': {c['name']: [] for c in conds},
                    'rerun_of': rerun_of, 'pause_after': pause_after, 'fly_age': self.t, 'started': time.time()}
        self.paused = False
        self.out_q.put({'type': 'experiment_started', 'id': eid, 'card': card, 'steps': EX.steps_text(card), 'rerun_of': rerun_of})

    def _exp_before_step(self):
        """Called once per loop iteration before the brain steps. Handles setup and settling."""
        e = self.exp
        if e is None: return
        if e['phase'] == 'setup':
            name, rep = e['plan'][e['i']]
            self.handle({'cmd': 'restore'}); self.handle({'cmd': 'reset_fly'})
            for a in e['conds'][name]['setup']:
                self.handle(dict(a))
            e['phase'], e['settle'] = 'settle', 30
        elif e['phase'] == 'settle':
            e['settle'] -= 1
            if e['settle'] <= 0:
                e['phase'] = 'run'; e['meter'] = self.ph.TrialMeter(self.brain, self.body)

    def _exp_after_step(self, spiked, motor, prev, caught_now):
        """Called after the body moved. Records the run; advances the plan; finishes."""
        e = self.exp
        if e is None or e['phase'] != 'run': return
        e['meter'].update(spiked, motor, self.body, prev, caught_now)
        if e['meter'].n >= e['card']['steps']:
            name, rep = e['plan'][e['i']]
            e['results'][name].append(e['meter'].result(self.brain, self.pharm))
            self.out_q.put({'type': 'experiment_run', 'id': e['id'], 'condition': name, 'repeat': rep + 1,
                            'value': EX.get_measure(e['results'][name][-1], e['card']['measure'])})
            e['i'] += 1
            if e['i'] >= len(e['plan']):
                self._exp_finish()
            else:
                e['phase'] = 'setup'

    def _exp_finish(self):
        e = self.exp; card = e['card']
        analysis, verdict, details = EX.analyse(card, e['results'])
        record = {'id': e['id'], 'card': card, 'results': e['results'], 'analysis': analysis, 'verdict': verdict,
                  'details': details, 'date': time.strftime('%Y-%m-%d %H:%M'), 'subject': getattr(self.brain, 'wiring', 'unknown'),
                  'fly_age': e['fly_age'], 'rerun_of': e['rerun_of'], 'seconds': round(time.time() - e['started'], 1)}
        self.notebook.save(record)
        try:
            with open(os.path.join(self.notebook.path, e['id'] + '.md'), 'w') as f:
                f.write(self.notebook.markdown(record))
        except Exception:
            pass
        self.out_q.put({'type': 'experiment_done', 'id': e['id'], 'record': record, 'steps': EX.steps_text(record['card']), 'notebook': self.notebook.list()})
        print(f"experiment [{card['title']}] -> {verdict}")
        self.handle({'cmd': 'restore'})
        if e['pause_after']: self.paused = True
        self.exp = None

    def _exp_state(self):
        e = self.exp
        if e is None: return None
        name, rep = e['plan'][e['i']] if e['i'] < len(e['plan']) else (None, 0)
        conds = list(e['results'])
        means = {c: (sum(EX.get_measure(r, e['card']['measure']) for r in v) / len(v)) if v else None for c, v in e['results'].items()}
        run_prog = (e['meter'].n / e['card']['steps']) if (e['phase'] == 'run' and e['meter']) else 0.0
        return {'id': e['id'], 'title': e['card']['title'], 'measure': e['card']['measure'],
                'condition': name, 'cond_i': conds.index(name) + 1 if name in conds else 0, 'n_cond': len(conds),
                'repeat': rep + 1, 'n_rep': e['card']['repeats'], 'phase': e['phase'],
                'progress': round((e['i'] + run_prog) / len(e['plan']), 3), 'means': means,
                'values': {c: [EX.get_measure(r, e['card']['measure']) for r in v] for c, v in e['results'].items()}}

    def _process_commands(self):
        while True:
            try:
                cmd = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            try:
                data = self.handle(cmd)
                self.out_q.put({'type': 'ack', 'id': cmd.get('id'), 'cmd': cmd.get('cmd'), 'ok': True,
                                'error': None, 'data': data})
            except Exception as e:
                self.out_q.put({'type': 'ack', 'id': cmd.get('id'), 'cmd': cmd.get('cmd'), 'ok': False,
                                'error': f"{type(e).__name__}: {e}", 'data': None})

    # ---- the loop --------------------------------------------------------------------------
    def run(self):
        b, w, body, pharm = self.brain, self.world, self.body, self.pharm
        t_win, n_win = time.time(), 0
        while self.running:
            self._process_commands()
            if self.paused:
                time.sleep(0.02); continue
            t_step = time.time()
            self._exp_before_step()

            sensory, dist = w.get_sensory(body.x, body.y, body.angle)
            caught_now = dist < w.catch_radius
            if caught_now:
                self.caught += 1
                w.randomize_sugar()
            motor, spiked = b.step(sensory)
            pharm.step()
            prev = (body.x, body.y)
            body.move(motor)
            self.t += 1
            self._exp_after_step(spiked, motor, prev, caught_now)

            if self.trial is not None:
                tr = self.trial
                tr['meter'].update(spiked, motor, body, prev, caught_now)
                if tr['meter'].n >= tr['steps']:
                    self.out_q.put({'type': 'result', 'id': tr['id'], 'label': tr['label'],
                                    'metrics': tr['meter'].result(b, pharm)})
                    self.trial = None
                    if tr['pause_after']:                  # freeze: experiment time = simulation time
                        self.paused = True

            if self.t % 25 == 0:
                r = b.rates(spiked)
                self.narrator.ask({'left': float(sensory[0]), 'right': float(sensory[1]), 'dist': dist,
                                   'rate_all': float(spiked.mean()), 'rate_L': r['motor_L'], 'rate_R': r['motor_R'],
                                   'forward': float(motor[0]), 'turn': float(motor[1]), 'n': b.n,
                                   'scrambled': b.scrambled, 'drugs': pharm.state()})

            n_win += 1
            if n_win >= 30:
                now = time.time(); self.sps = n_win / (now - t_win); t_win, n_win = now, 0
            self._snapshot(spiked, motor, sensory, dist)

            if self.sps_cap > 0:
                dt = time.time() - t_step
                if dt < 1.0 / self.sps_cap:
                    time.sleep(1.0 / self.sps_cap - dt)

    def _snapshot(self, spiked, motor, sensory, dist):
        b, w, body, pharm = self.brain, self.world, self.body, self.pharm
        r = b.rates(spiked); r['all'] = float(spiked.mean())
        tr = self.trial
        with self.lock:
            self.fired_union |= spiked[self.sample_idx]
            self.snap = {'type': 'state', 't': self.t, 'sps': round(self.sps, 1),
                         'fly': {'x': body.x, 'y': body.y, 'angle': float(body.angle)},
                         'banana': {'x': w.sugar_x, 'y': w.sugar_y}, 'dist': float(dist), 'caught': self.caught,
                         'motor': {'forward': float(motor[0]), 'turn': float(motor[1])},
                         'smell': {'left': float(sensory[0]), 'right': float(sensory[1])},
                         'rates': {k: round(float(v), 4) for k, v in r.items()},
                         'scrambled': bool(b.scrambled), 'drugs': pharm.state(),
                         'params': {k: round(float(v), 4) for k, v in b.params.items()},
                         'thought': self.narrator.get(), 'paused': self.paused,
                         'trial': None if tr is None else {'id': tr['id'], 'label': tr['label'], 'steps': tr['steps'],
                                                           'progress': round(tr['meter'].n / tr['steps'], 3)},
                         'experiment': self._exp_state()}

    def take_state(self):
        """Called by the broadcaster: latest snapshot plus the union of spikes since last call."""
        with self.lock:
            fired = np.flatnonzero(self.fired_union).tolist()
            self.fired_union[:] = False
            st = dict(self.snap)
        st['fired'] = fired
        return st


# =========================================================================================
# networking (shared with fake_server.py)
# =========================================================================================
def serve(sim, ws_port=WS_PORT, http_port=HTTP_PORT, static_dir=None):
    static_dir = static_dir or os.path.dirname(os.path.abspath(__file__))
    clients = set()
    from collections import deque
    recent_logs = deque(maxlen=30)                     # replayed to clients that connect late

    # -- static files for lab.html ------------------------------------------------------------
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=static_dir, **k)
        def log_message(self, *a):
            pass
        def end_headers(self):
            self.send_header('Cache-Control', 'no-store'); super().end_headers()
    httpd = http.server.ThreadingHTTPServer(('0.0.0.0', http_port), Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    threading.Thread(target=sim.run, daemon=True).start()

    async def send_all(msg):
        if not clients:
            return
        data = json.dumps(msg)
        dead = []
        for ws in list(clients):
            try:
                await ws.send(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    async def handler(ws, *_):
        clients.add(ws)
        try:
            await ws.send(json.dumps(sim.hello()))
            for msg in list(recent_logs):
                await ws.send(json.dumps(msg))
            async for raw in ws:
                try:
                    cmd = json.loads(raw)
                except Exception:
                    await ws.send(json.dumps({'type': 'ack', 'id': None, 'cmd': None, 'ok': False,
                                              'error': 'not JSON', 'data': None}))
                    continue
                if cmd.get('cmd') == 'log':
                    msg = {'type': 'log', 'source': str(cmd.get('source', 'client')),
                           'text': str(cmd.get('text', ''))[:500], 'ts': time.time()}
                    recent_logs.append(msg)
                    await send_all(msg)
                    continue
                if cmd.get('cmd') == 'lookup':                 # network lookups run off the sim thread
                    asyncio.create_task(do_lookup(ws, cmd))
                    continue
                sim.cmd_q.put(cmd)
        finally:
            clients.discard(ws)

    async def do_lookup(ws, cmd):
        import substances as SUB
        name = str(cmd.get('name', '')).strip()
        if not name:
            await ws.send(json.dumps({'type': 'ack', 'id': cmd.get('id'), 'cmd': 'lookup', 'ok': False, 'error': 'lookup needs a name', 'data': None})); return
        r = await asyncio.to_thread(SUB.lookup, name, sim.ph.DRUGS, not cmd.get('offline', False))
        m = r.get('match')
        text = f"looked up '{name}': " + (f"{m['name']} ({m.get('confidence', '')})" if m else 'not in the library') + \
               "; " + ", ".join(s.get('source', '?') + ('' if 'error' not in s else ' unavailable') for s in r.get('sources', []))
        msg = {'type': 'log', 'source': 'lab', 'text': text, 'ts': time.time()}
        recent_logs.append(msg); await send_all(msg)
        await ws.send(json.dumps({'type': 'ack', 'id': cmd.get('id'), 'cmd': 'lookup', 'ok': True, 'error': None, 'data': r}))

    async def broadcaster():
        period = 1.0 / BROADCAST_HZ
        while True:
            await asyncio.sleep(period)
            while True:                                    # acks, results, logs from the sim thread
                try:
                    msg = sim.out_q.get_nowait()
                except queue.Empty:
                    break
                if msg.get('type') in ('experiment_started', 'experiment_run'):
                    pass
                if msg.get('type') == 'result':
                    print(f"result [{msg.get('label')}]: " + ", ".join(
                        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in msg['metrics'].items() if k not in ('firing', 'drugs')))
                await send_all(msg)
            if clients:
                await send_all(sim.take_state())

    async def main():
        async with websockets.serve(handler, '0.0.0.0', ws_port, max_size=2 ** 20):
            print(f"Lab server: ws://localhost:{ws_port}   files: http://localhost:{http_port}/lab.html")
            await broadcaster()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        sim.running = False
        httpd.shutdown()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--sps', type=float, default=60, help='cap on brain steps per second (0 = unlimited)')
    ap.add_argument('--no-narrator', action='store_true')
    ap.add_argument('--brain', choices=['flywire', 'random'], default='flywire',
                    help='flywire = real FlyWire v783 wiring (needs flywire/maxime_v783.npz); random = synthetic fly')
    ap.add_argument('--ws-port', type=int, default=WS_PORT)
    ap.add_argument('--http-port', type=int, default=HTTP_PORT)
    a = ap.parse_args()
    serve(Sim(seed=a.seed, sps_cap=a.sps, narrate=not a.no_narrator, brain=a.brain), a.ws_port, a.http_port)
