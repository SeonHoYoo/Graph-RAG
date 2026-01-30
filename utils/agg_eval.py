from collections import defaultdict
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    help="json file path containing predicted and ground truth answers together.",
)
args = parser.parse_args()


with open(args.input_path, 'r') as f:
    data = json.load(f)

em_scores = defaultdict(list)
f1_scores = defaultdict(list)   

for sample in data:
    em_scores['total'].append(sample['em_score'] or 0)
    em_scores[str(sample['num_hops'])].append(sample['em_score'] or 0)

    f1_scores['total'].append(sample['f1_score'] or 0)
    f1_scores[str(sample['num_hops'])].append(sample['f1_score'] or 0)

keys = list(em_scores.keys())
keys.sort()

for key in keys:
    em_avg = sum(em_scores[key]) / len(em_scores[key]) if em_scores[key] else 0
    f1_avg = sum(f1_scores[key]) / len(f1_scores[key]) if f1_scores[key] else 0
    
    with open(args.input_path.replace(".json", ".out"), 'a') as f:
        f.write(f'Hops: {key}, EM: {em_avg:.3f}, F1: {f1_avg:.3f}\n')
