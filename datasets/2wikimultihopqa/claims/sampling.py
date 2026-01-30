import json
import random

split = 'train'

with open(f'{split}.json') as f:
    data = json.load(f)

num_samples = 500
sampled_total = random.sample(data, num_samples)

for i, item in enumerate(sampled_total):
    item["index"] = i

unsampled = [item for item in data if item not in sampled_total]
for i, item in enumerate(unsampled):  
    item["index"] = i 


save_fname = f'{split}_sampled.json'
save_fname_unsampled = f'{split}_unsampled.json'

with open(save_fname, 'w') as f:
    json.dump(sampled_total, f, indent=4)
with open(save_fname_unsampled, 'w') as f:
    json.dump(unsampled, f, indent=4)