import os
import json
import collections
import matplotlib.pyplot as plt
import numpy as np
import argparse
import statistics

def analyze_semantic_relations(directory):
    total_edge_counts = collections.Counter()
    total_edge_type_counts = collections.Counter()
    total_triplet_counts = collections.Counter()
    total_triplet_comb_counts = collections.Counter()
    total_triplet_streaks = collections.defaultdict(list)
    total_triplet_comb_streaks = collections.defaultdict(list)
    file_results = {}
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            edge_counts = collections.Counter()
            edge_type_counts = collections.Counter()
            triplet_counts = collections.Counter()
            triplet_comb_counts = collections.Counter()
            triplet_streaks = collections.defaultdict(list)
            triplet_comb_streaks = collections.defaultdict(list)
            
            previous_triplets = set()
            current_streaks = collections.defaultdict(int)
            comb_streak = 0
            
            for obj_id, obj_data in data.items():
                edges = obj_data.get("semantic_relations", [])
                edge_counts[len(edges)] += 1
                
                triplet_set = set()
                for v1, edge, v2 in edges:
                    edge_type_counts[edge] += 1
                    triplet_counts[(v1, edge, v2)] += 1
                    triplet_set.add((v1, edge, v2))
                
                if triplet_set != set():
                    triplet_comb_counts[frozenset(triplet_set)] += 1
                    if triplet_set == previous_triplets:
                        comb_streak += 1
                        for triplet in triplet_set:
                            current_streaks[triplet] += 1
                    else:
                        triplet_comb_streaks[frozenset(previous_triplets)].append(comb_streak)
                        comb_streak = 1
                        for triplet in triplet_set & previous_triplets:
                            current_streaks[triplet] += 1
                        for triplet in triplet_set - previous_triplets:
                            current_streaks[triplet] = 1
                        for triplet in previous_triplets - triplet_set:
                            triplet_streaks[triplet].append(current_streaks[triplet])
                            del current_streaks[triplet]
                elif previous_triplets != set():
                    for triplet, streak in current_streaks.items():
                        triplet_streaks[triplet].append(streak)
                    current_streaks = collections.defaultdict(int)
                    triplet_comb_streaks[frozenset(previous_triplets)].append(comb_streak)
                    
                previous_triplets = triplet_set
                
            file_results[filename] = {
                "edge_counts": dict(edge_counts),
                "edge_type_counts": dict(edge_type_counts),
                "triplet_counts": dict(triplet_counts),
                "triplet_comb_counts": dict(triplet_comb_counts),
                "triplet_streaks": triplet_streaks,
                "triplet_comb_streaks": triplet_comb_streaks,
            }
            
            total_edge_counts.update(edge_counts)
            total_edge_type_counts.update(edge_type_counts)
            total_triplet_counts.update(triplet_counts)
            total_triplet_comb_counts.update(triplet_comb_counts)
            for key, value in triplet_streaks.items():
                total_triplet_streaks[key].extend(value)
            for key, value in triplet_comb_streaks.items():
                total_triplet_comb_streaks[key].extend(value)
    
    total_results = {
        "edge_counts": dict(total_edge_counts),
        "edge_type_counts": dict(total_edge_type_counts),
        "triplet_counts": dict(total_triplet_counts),
        "triplet_comb_counts": dict(total_triplet_comb_counts),
        "triplet_streaks": total_triplet_streaks,
        "triplet_comb_streaks": total_triplet_comb_streaks,
    }
    
    return file_results, total_results

def write_analysis_to_file(file_results, total_results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Per File Analysis:\n")
        for filename in sorted(file_results.keys()):
            results = file_results[filename]
            total_frames = sum(results["edge_counts"].values())
            f.write(f"\n{filename}:\n")
            
            f.write("Frames with x edges:\n")
            for key, count in sorted(results["edge_counts"].items()):
                f.write(f"  {key}: {count} ({(count / total_frames) * 100:.2f}%)\n")
            f.write(f"Frames Total: {total_frames}\n")
            
            sorted_edge_types = sorted(results["edge_type_counts"].items(), key=lambda x: x[1], reverse=True)
            for edge, count in sorted_edge_types:
                percentage = (count / total_frames) * 100
                f.write(f"  {edge}: {count} ({percentage:.2f}%)\n")
            
            f.write("Top 10 highest average streak edge triplets:\n")
            max_avg_streak_triplets = sorted(results["triplet_streaks"], key=lambda k: statistics.mean(results["triplet_streaks"][k]), reverse=True)[:10]
            for (v1, edge, v2) in max_avg_streak_triplets:
                streak = results['triplet_streaks'][(v1, edge, v2)]
                non_one_streaks = sum(1 for num in streak if num > 1)
                avg = round(sum(streak) / len(streak), 2)
                std_dev = round(statistics.stdev(streak), 2) if len(streak) > 1 else 0
                median = statistics.median(streak)
                max_value = max(streak)
                percentage = (results["triplet_counts"][(v1, edge, v2)] / total_frames) * 100
                
                if std_dev > 0:
                    f.write(
                        f"{f'{v1} - {edge} - {v2}:':<46} {results['triplet_counts'][(v1, edge, v2)]:>5} "
                        f"({percentage:5.2f}%); Streak Info:  "
                        f"Streak_Num: {len(streak):>3}  "
                        f"Num_of_streaks>1: {non_one_streaks:>3}  "
                        f"Avg: {avg:>7}  "
                        f"Std_Dev: {std_dev:>7}  "
                        f"Median: {median:>7}  "
                        f"Max: {max_value:>7}\n")
                else:
                    f.write(
                        f"{f'{v1} - {edge} - {v2}:':<46} {results['triplet_counts'][(v1, edge, v2)]:>5} "
                        f"({percentage:5.2f}%); Streak Info:  "
                        f"Streak_Num: {len(streak):>3}  "
                        f"Num_of_streaks>1: {non_one_streaks:>3}\n")

        f.write("\nTotal Analysis:\n")
        total_frames = sum(total_results["edge_counts"].values())
        
        f.write("Frames with x edges:\n")
        for key, count in sorted(total_results["edge_counts"].items()):
            f.write(f"  {key}: {count} ({(count / total_frames) * 100:.2f}%)\n")
        f.write(f"Frames Total: {total_frames}\n")
        
        sorted_edge_types = sorted(total_results["edge_type_counts"].items(), key=lambda x: x[1], reverse=True)
        for edge, count in sorted_edge_types:
            percentage = (count / total_frames) * 100
            f.write(f"  {edge}: {count} ({percentage:.2f}%)\n")
        
        f.write("\nTotal triplet counts and streak statistics:\n")
        sorted_triplets = sorted(total_results["triplet_counts"].items(), key=lambda x: x[1], reverse=True)
        for (v1, edge, v2), count in sorted_triplets:
            streak = total_results['triplet_streaks'][(v1, edge, v2)]
            non_one_streaks = sum(1 for num in streak if num > 1)
            avg = round(sum(streak) / len(streak), 2)
            std_dev = round(statistics.stdev(streak), 2) if len(streak) > 1 else 0
            median = statistics.median(streak)
            max_value = max(streak)
            percentage = (count / total_frames) * 100
            
            if std_dev > 0:
                f.write(
                    f"{f'{v1} - {edge} - {v2}:':<46} {count:>5} "
                    f"({percentage:5.2f}%); Streak Info:  "
                    f"Streak_Num: {len(streak):>3}  "
                    f"Num_of_streaks>1: {non_one_streaks:>3}  "
                    f"Avg: {avg:>7}  "
                    f"Std_Dev: {std_dev:>7}  "
                    f"Median: {median:>7}  "
                    f"Max: {max_value:>7}\n")
            else:
                f.write(
                    f"{f'{v1} - {edge} - {v2}:':<46} {count:>5} "
                    f"({percentage:5.2f}%); Streak Info:  "
                    f"Streak_Num: {len(streak):>3}  "
                    f"Num_of_streaks>1: {non_one_streaks:>3}\n")
        
        f.write("\nTop 20 highest average streaks edge triplet combinations:\n")
        max_avg_streak_triplet_combs = sorted(total_results["triplet_comb_streaks"], key=lambda k: statistics.mean(total_results["triplet_comb_streaks"][k]), reverse=True)[:20]
        for edge_set in max_avg_streak_triplet_combs:
            streak = total_results['triplet_comb_streaks'][edge_set]
            avg = round(sum(streak) / len(streak), 2)
            std_dev = round(statistics.stdev(streak), 2) if len(streak) > 1 else 0
            median = statistics.median(streak)
            max_value = max(streak)
            
            f.write("\n")
            for (v1, edge, v2) in edge_set:
                f.write(f"  {v1} - {edge} - {v2}\n")
            
            if std_dev > 0:
                f.write(f"  Count: {total_results['triplet_comb_counts'][edge_set]}; Streak Info: Avg: {avg}\tStd_Dev: {std_dev}\tMedian: {median}\tMax: {max_value}\n")
            else:
                f.write(f"  Count: {total_results['triplet_comb_counts'][edge_set]}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze semantic relations in JSON files.")
    parser.add_argument("directory", type=str, help="Path to the directory containing JSON files.")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "graphs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.txt")
    
    file_results, total_results = analyze_semantic_relations(args.directory)
    write_analysis_to_file(file_results, total_results, output_file)

if __name__ == "__main__":
    main()
