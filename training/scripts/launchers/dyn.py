import json, sys
x = sys.argv[1]; m = '/mnt/data/peilin/sigma-mem/outputs/rl/%s/metrics.jsonl' % x
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
try:
    rows = [json.loads(l) for l in open(m) if l.strip()]
except FileNotFoundError:
    sys.exit(0)
W = {'rag': 231, 'math': 162, 'code': 119}
for r in rows[-n:]:
    s = int(r.get('training/global_step', -1))
    g = lambda k, d=3: (round(r[k], d) if k in r else None)
    val = {t: r.get('val-core/%s/acc/mean@1' % t) for t in W if ('val-core/%s/acc/mean@1' % t) in r}
    vtxt = ''
    if val:
        w = sum(W[t] for t in val)
        vtxt = ' VAL=%.1f (rag %.1f math %.1f code %.1f)' % (100 * sum(W[t] * v for t, v in val.items()) / w, 100 * val.get('rag', 0), 100 * val.get('math', 0), 100 * val.get('code', 0))
    if 'critic/score/mean' in r:
        print('%s step %d: reward %s kl %s entropy %s grad %s len %s trunc %s zero_groups %s t %ss%s' % (x, s, g('critic/score/mean'), g('actor/kl_loss', 4), g('actor/entropy_loss'), g('actor/grad_norm', 2), g('response_length/mean', 0), g('response_length/clip_ratio'), g('memory/zero_groups', 0), g('timing_s/step', 0), vtxt))
    elif vtxt:
        print('%s step %d:%s' % (x, s, vtxt))
