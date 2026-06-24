import json
p = 'C:/Development/python/model-training/LemGendaryDatasets/LemGendizedNimaAestheticMobileLarge/index.json'
with open(p) as f:
    d = json.load(f)

for i in range(5):
    print(f"{d[i].get('source')} | {d[i].get('name')} | Score: {d[i].get('nima_score')}")

import random
sample = random.sample(d, 5)
for s in sample:
    print(f"RND: {s.get('source')} | {s.get('name')} | Score: {s.get('nima_score')}")
