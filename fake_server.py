"""
fake_server.py  -  speaks protocol.md with a SCRIPTED fly and no brain.

    python3 fake_server.py

Use it to build or demo lab.html without the simulation (no numpy brain, starts instantly).
Everything it reports is made up and labelled mode="fake" in the hello message.
"""
import argparse, math, queue, random, threading, time
import server as S


class FakeSim:
    mode = 'fake'

    def __init__(self, sps_cap=60):
        self.n, self.arena, self.catch_radius = 139255, 10.0, 0.9
        self.regions = {'sens_L': [0, 5000], 'sens_R': [5000, 10000], 'central': [10000, 120000],
                        'motor_L': [120000, 129500], 'motor_R': [129500, 139255]}
        self.sample_size = sum(c for _, c in S.SAMPLE_BANDS)
        self.params = {'I_bg': 1.3, 'g_exc': 0.5, 'g_inh': 0.5, 'ttx_frac': 0.0, 'motor_out': 1.0, 'turn_gain': 15.0}
        self.drugs_info = {'picrotoxin': 400, 'ttx': 800, 'tetanus_toxin': 2000, 'caffeine': 600,
                           'isoflurane': 120, 'ethanol': 600, 'nicotine': 300}
        self.x, self.y, self.angle = 0.0, 0.0, 0.3
        self.bx, self.by = 4.0, 2.0
        self.t, self.caught, self.sps, self.sps_cap = 0, 0, 0.0, sps_cap
        self.scrambled, self.paused, self.running = False, False, True
        self.drugs, self.trial = {}, None
        self.lock = threading.Lock()
        self.cmd_q, self.out_q = queue.Queue(), queue.Queue()
        self.fired = set()
        self.snap = {}
        self.thought = "(fake) scripted fly - no brain connected"
        self._snapshot()

    def hello(self):
        return {'type': 'hello', 'mode': 'fake', 'name': 'Maxime (fake)', 'n': self.n, 'arena': self.arena, 'catch_radius': self.catch_radius,
                'regions': self.regions,
                'sample': {'size': self.sample_size, 'bands': [list(b) for b in S.SAMPLE_BANDS]},
                'params': dict(self.params),
                'drugs': {k: {'mechanism': 'fake', 'half_life': v} for k, v in self.drugs_info.items()}}

    def handle(self, cmd):
        c = cmd.get('cmd')
        if c == 'ping': return {}
        if c == 'scramble': self.scrambled = True; return {'scrambled': True}
        if c == 'restore': self.scrambled = False; self.drugs.clear(); return {'scrambled': False}
        if c == 'reset_fly':
            self.x, self.y, self.angle = 0.0, 0.0, random.uniform(-math.pi, math.pi); self._new_banana()
            return {'x': 0.0, 'y': 0.0, 'angle': self.angle}
        if c == 'move_banana':
            if 'x' in cmd and 'y' in cmd:
                self.bx, self.by = max(-9, min(9, float(cmd['x']))), max(-9, min(9, float(cmd['y'])))
            else:
                self._new_banana()
            return {'x': self.bx, 'y': self.by}
        if c == 'dose':
            d = str(cmd['drug'])
            if d not in self.drugs_info: raise KeyError(f"unknown drug '{d}'")
            self.drugs[d] = min(2.0, self.drugs.get(d, 0.0) + float(cmd['amount'])); return {d: self.drugs[d]}
        if c == 'washout': self.drugs.clear(); return {'drugs': {}}
        if c == 'set_param':
            self.params[str(cmd['key'])] = float(cmd['value']); return {cmd['key']: self.params[cmd['key']]}
        if c == 'get_params': return {'params': dict(self.params), 'drugs': dict(self.drugs)}
        if c == 'run_trial':
            if self.trial: raise RuntimeError('a trial is already running')
            steps = int(cmd.get('steps', 400))
            self.trial = {'id': cmd.get('id'), 'label': str(cmd.get('label', '')), 'steps': steps, 'n': 0, 'caught0': self.caught,
                          'pause_after': bool(cmd.get('pause_after', False))}
            self.paused = False
            return {'steps': steps}
        if c == 'cancel_trial': self.trial = None; return {}
        if c == 'speed': self.sps_cap = float(cmd.get('sps', 60)); return {'sps': self.sps_cap}
        if c == 'pause': self.paused = True; return {'paused': True}
        if c == 'resume': self.paused = False; return {'paused': False}
        raise ValueError(f"unknown cmd '{c}'")

    def _new_banana(self):
        self.bx, self.by = random.uniform(-6, 6), random.uniform(-6, 6)

    def _process_commands(self):
        while True:
            try: cmd = self.cmd_q.get_nowait()
            except queue.Empty: return
            try:
                data = self.handle(cmd)
                self.out_q.put({'type': 'ack', 'id': cmd.get('id'), 'cmd': cmd.get('cmd'), 'ok': True, 'error': None, 'data': data})
            except Exception as e:
                self.out_q.put({'type': 'ack', 'id': cmd.get('id'), 'cmd': cmd.get('cmd'), 'ok': False,
                                'error': f"{type(e).__name__}: {e}", 'data': None})

    def run(self):
        t_win, n_win = time.time(), 0
        while self.running:
            self._process_commands()
            if self.paused:
                time.sleep(0.02); continue
            t0 = time.time()
            # scripted behaviour: steer toward the banana unless scrambled; slow down under drugs
            slow = 1.0 - 0.9 * min(1.0, self.drugs.get('ttx', 0) + self.drugs.get('tetanus_toxin', 0) + 0.7 * self.drugs.get('isoflurane', 0))
            dx, dy = self.bx - self.x, self.by - self.y
            dist = math.hypot(dx, dy)
            want = math.atan2(dy, dx)
            err = (want - self.angle + math.pi) % (2 * math.pi) - math.pi
            turn = (0.0 if self.scrambled else 0.12 * err) + random.gauss(0, 0.06)
            self.angle += turn
            fwd = 0.09 * slow
            self.x += fwd * math.cos(self.angle); self.y += fwd * math.sin(self.angle)
            if abs(self.x) > 9.9: self.angle = math.pi - self.angle; self.x = max(-9.9, min(9.9, self.x))
            if abs(self.y) > 9.9: self.angle = -self.angle; self.y = max(-9.9, min(9.9, self.y))
            if dist < self.catch_radius:
                self.caught += 1; self._new_banana()
            for k in list(self.drugs):
                self.drugs[k] *= 0.5 ** (1.0 / self.drugs_info[k])
                if self.drugs[k] < 1e-3: del self.drugs[k]
            self.t += 1
            base = 0.045 * slow
            k = int(self.sample_size * base * random.uniform(0.7, 1.3))
            with self.lock:
                self.fired.update(random.sample(range(self.sample_size), k))
            if self.trial:
                self.trial['n'] += 1
                if self.trial['n'] >= self.trial['steps']:
                    tr = self.trial; self.trial = None
                    if tr['pause_after']: self.paused = True
                    self.out_q.put({'type': 'result', 'id': tr['id'], 'label': tr['label'], 'metrics': {
                        'steps': tr['steps'], 'bananas': self.caught - tr['caught0'], 'activity': fwd,
                        'tortuosity': 2.0, 'mean_abs_turn': 0.3, 'firing_all': base, 'firing_peak': base * 1.3,
                        'firing': {r: base for r in self.regions}, 'synchrony': 0.0,
                        'drugs': dict(self.drugs), 'scrambled': self.scrambled, 'fake': True}})
            n_win += 1
            if n_win >= 30:
                now = time.time(); self.sps = n_win / (now - t_win); t_win, n_win = now, 0
            self._snapshot(dist, fwd, turn, base)
            dt = time.time() - t0
            cap = self.sps_cap if self.sps_cap > 0 else 1000.0      # "unlimited" = 1000/s here (no brain to slow it)
            if dt < 1 / cap:
                time.sleep(1 / cap - dt)

    def _snapshot(self, dist=5.0, fwd=0.0, turn=0.0, base=0.045):
        tr = self.trial
        side = math.sin(math.atan2(self.by - self.y, self.bx - self.x) - self.angle)
        with self.lock:
            self.snap = {'type': 'state', 't': self.t, 'sps': round(self.sps, 1),
                         'fly': {'x': self.x, 'y': self.y, 'angle': self.angle},
                         'banana': {'x': self.bx, 'y': self.by}, 'dist': dist, 'caught': self.caught,
                         'motor': {'forward': fwd, 'turn': turn},
                         'smell': {'left': max(0, 0.5 + 0.3 * side), 'right': max(0, 0.5 - 0.3 * side)},
                         'rates': {'sens_L': base * (1.4 + side), 'sens_R': base * (1.4 - side), 'central': base,
                                   'motor_L': base * (1 + 0.5 * side), 'motor_R': base * (1 - 0.5 * side), 'all': base},
                         'scrambled': self.scrambled, 'drugs': {k: round(v, 4) for k, v in self.drugs.items()},
                         'params': dict(self.params), 'thought': self.thought, 'paused': self.paused,
                         'trial': None if tr is None else {'id': tr['id'], 'label': tr['label'], 'steps': tr['steps'],
                                                           'progress': round(tr['n'] / tr['steps'], 3)}}

    def take_state(self):
        with self.lock:
            st = dict(self.snap); st['fired'] = sorted(self.fired); self.fired.clear()
        return st


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ws-port', type=int, default=S.WS_PORT)
    ap.add_argument('--http-port', type=int, default=S.HTTP_PORT)
    a = ap.parse_args()
    print("FAKE server: scripted fly, no brain. For the real thing run server.py")
    S.serve(FakeSim(), a.ws_port, a.http_port)
