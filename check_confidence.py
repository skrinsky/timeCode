import json

confs = [json.loads(l)['adsr_confidence'] for l in open('data/nsynth_estimated/metadata.jsonl')]
print(f'total clips: {len(confs)}')
print(f'mean: {sum(confs)/len(confs):.3f}, max: {max(confs):.3f}, min: {min(confs):.3f}')
confs.sort(reverse=True)
print(f'top 10: {confs[:10]}')
print(f'above 0.5: {sum(1 for c in confs if c >= 0.5)}')
print(f'above 0.2: {sum(1 for c in confs if c >= 0.2)}')
print(f'above 0.1: {sum(1 for c in confs if c >= 0.1)}')
