import json
import random
import os

def main():
    input_file = "dataset/test_set_boosted.json" # try to use balanced or boosted
    if not os.path.exists(input_file):
        input_file = "dataset/test_set_balanced.json"
        
    output_file = "dataset/test_set_1000.json"
    
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    pos_samples = [d for d in data if d.get('label') == 1]
    neg_samples = [d for d in data if d.get('label') == 0]
    
    print(f"Found {len(pos_samples)} positives and {len(neg_samples)} negatives.")
    
    disease_pos_counts = {}
    disease_neg_counts = {}
    for p in pos_samples:
        disease_pos_counts[p['disease']] = disease_pos_counts.get(p['disease'], 0) + 1
    for n in neg_samples:
        disease_neg_counts[n['disease']] = disease_neg_counts.get(n['disease'], 0) + 1

    def sample_proportionally(samples, category_counts, target_total):
        sampled = []
        total_in_population = sum(category_counts.values())
        
        categorized_samples = {}
        for s in samples:
            d = s['disease']
            if d not in categorized_samples:
                categorized_samples[d] = []
            categorized_samples[d].append(s)
            
        remaining_target = target_total
        for d, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            if remaining_target <= 0:
                break
                
            proportion = count / total_in_population
            quota = min(int(round(proportion * target_total)), remaining_target, len(categorized_samples[d]))
            
            if quota == 0 and remaining_target > 0 and len(categorized_samples[d]) > 0:
                 quota = 1
                 
            if quota > 0:
                sampled.extend(random.sample(categorized_samples[d], quota))
                remaining_target -= quota

        if len(sampled) < target_total:
            sampled_ids = {f"{s['mirna']}_{s['disease']}" for s in sampled}
            remaining_pool = [s for s in samples if f"{s['mirna']}_{s['disease']}" not in sampled_ids]
            needed = target_total - len(sampled)
            sampled.extend(random.sample(remaining_pool, min(needed, len(remaining_pool))))
            
        return sampled

    random.seed(42) # For reproducibility
    print("Sampling proportionally to match disease distribution in original dataset...")
    pos_subset = sample_proportionally(pos_samples, disease_pos_counts, 500)
    neg_subset = sample_proportionally(neg_samples, disease_neg_counts, 500)
    
    subset = pos_subset + neg_subset
    random.shuffle(subset)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(subset)} samples ({len(pos_subset)} pos, {len(neg_subset)} neg) to {output_file}")

if __name__ == "__main__":
    main()
