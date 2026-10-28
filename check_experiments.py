"""
check_experiments.py - runs one real experiment card through a running server and checks the
notebook machinery: proposal -> run -> live progress -> verdict -> saved record -> rerun.
    python3 server.py --brain random      (random is faster for this check)
    python3 check_experiments.py
"""
import asyncio, json, sys, time
import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else 'ws://localhost:8765'
R = {}
def ok(name, cond, note=''):
    R[name] = bool(cond); print(f"   {'PASS' if cond else 'FAIL'}  {name}  {note}")

CARD = {"title": "Does tetanus toxin stop Maxime walking?",
        "hypothesis": "Tetanus toxin blocks the leg junctions, so Maxime walks much less.",
        "description": "Tetanus toxin cleaves synaptobrevin at motor terminals; the brain should keep firing while the body stops.",
        "control": {"name": "sober", "setup": []},
        "treatments": [{"name": "tetanus 1.0", "setup": [{"cmd": "dose", "drug": "tetanus_toxin", "amount": 1.0}]}],
        "measure": "activity", "prediction": "lower", "repeats": 3, "steps": 120, "author": "check"}

async def main():
    async with websockets.connect(URL, max_size=2 ** 20) as ws:
        hello = json.loads(await ws.recv())
        ok('hello_has_measures_and_notebook', 'measures' in hello and 'notebook' in hello)
        n0 = len(hello['notebook'])
        pending, events = {}, []
        counter = [0]
        async def send(**kw):
            counter[0] += 1; cid = f"x{counter[0]}"; await ws.send(json.dumps(dict(id=cid, **kw))); return cid
        async def wait(pred, timeout=300):
            t0 = time.time()
            while time.time() - t0 < timeout:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout)); events.append(m)
                if pred(m): return m
            raise TimeoutError
        await send(cmd='speed', sps=0)
        # bad card rejected
        cid = await send(cmd='propose_experiment', card={"title": "x", "hypothesis": "y", "treatments": [{"name": "t", "setup": [{"cmd": "dose", "drug": "nope", "amount": 1}]}], "measure": "activity", "prediction": "higher"})
        a = await wait(lambda m: m.get('type') == 'ack' and m.get('id') == cid)
        ok('bad_card_rejected', not a['ok'], a.get('error', '')[:60])
        # propose a good one
        cid = await send(cmd='propose_experiment', card=CARD)
        a = await wait(lambda m: m.get('type') == 'ack' and m.get('id') == cid)
        ok('proposal_accepted', a['ok'] and 'steps' in a['data'], (a.get('error') or '')[:60])
        eid = a['data']['id']
        print("   steps:", "\n          ".join(a['data']['steps']))
        p = next((m for m in events if m.get('type') == 'proposal' and m.get('id') == eid), None)
        ok('proposal_broadcast', p is not None)
        # run it
        cid = await send(cmd='run_experiment', exp=eid)
        a = await wait(lambda m: m.get('type') == 'ack' and m.get('id') == cid)
        ok('run_accepted', a['ok'], (a.get('error') or ''))
        st = await wait(lambda m: m.get('type') == 'state' and m.get('experiment') and m['experiment']['phase'] == 'run', 60)
        ok('live_progress_in_state', st['experiment']['condition'] in ('sober', 'tetanus 1.0'), f"{st['experiment']['condition']} rep {st['experiment']['repeat']}")
        runs = []
        done = None
        t0 = time.time()
        while time.time() - t0 < 600:
            m = json.loads(await asyncio.wait_for(ws.recv(), 600))
            if m.get('type') == 'experiment_run': runs.append(m)
            if m.get('type') == 'experiment_done' and m.get('id') == eid: done = m; break
        ok('six_runs_reported', len(runs) == 6, f"{len(runs)} runs")
        ok('experiment_done', done is not None)
        rec = done['record']
        print("   verdict:", rec['verdict']); print("   analysis:", rec['analysis'].replace("\n", "\n             "))
        ok('verdict_supported', rec['verdict'] == 'SUPPORTED')
        ok('notebook_grew', len(done['notebook']) == n0 + 1)
        cid = await send(cmd='get_experiment', exp=eid)
        a = await wait(lambda m: m.get('type') == 'ack' and m.get('id') == cid)
        ok('record_retrievable_with_markdown', a['ok'] and '## Verdict' in a['data']['markdown'])
        # rerun
        cid = await send(cmd='run_experiment', exp=eid)
        a = await wait(lambda m: m.get('type') == 'ack' and m.get('id') == cid)
        ok('rerun_accepted', a['ok'] and a['data']['id'] != eid)
        done2 = await wait(lambda m: m.get('type') == 'experiment_done', 600)
        ok('rerun_links_to_original', done2['record'].get('rerun_of') == eid)
        await send(cmd='speed', sps=60)
    print("\n==== SUMMARY ====")
    for k, v in R.items(): print(f"   {k:34s} {'PASS' if v else 'FAIL'}")
    print("ALL PASS" if all(R.values()) else "NOT YET")

asyncio.run(main())
