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
                
                if triplet_set:
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
                elif previous_triplets:
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
            
            f.write("Top 10 highest average streak edge triplets (only streaks > 1 are considered):\n")
            filtered_triplets = []
            for triplet, streak in results["triplet_streaks"].items():
                filtered = [x for x in streak if x > 1]
                if filtered:
                    filtered_triplets.append((triplet, filtered))
            filtered_triplets = sorted(filtered_triplets, key=lambda t: statistics.mean(t[1]), reverse=True)[:10]
            for (v1, edge, v2), filtered in filtered_triplets:
                non_one_streaks = len(filtered)
                avg_val = round(sum(filtered)/len(filtered), 2)
                std_dev = round(statistics.stdev(filtered), 2) if len(filtered) > 1 else 0
                median_val = statistics.median(filtered)
                max_value = max(filtered)
                percentage = (results["triplet_counts"][(v1, edge, v2)] / total_frames) * 100
                f.write(
                    f"{f'{v1} - {edge} - {v2}:':<46} {results['triplet_counts'][(v1, edge, v2)]:>5} "
                    f"({percentage:5.2f}%); Streak Info:  "
                    f"Streak_Num: {non_one_streaks:>3}  "
                    f"Avg: {avg_val:>7}  "
                    f"Std_Dev: {std_dev:>7}  "
                    f"Median: {median_val:>7}  "
                    f"Max: {max_value:>7}\n")
        
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
        
        f.write("\nTotal triplet counts and streak statistics (only streaks > 1 are considered):\n")
        sorted_triplets = sorted(total_results["triplet_counts"].items(), key=lambda x: x[1], reverse=True)
        for (v1, edge, v2), count in sorted_triplets:
            streak = total_results['triplet_streaks'][(v1, edge, v2)]
            filtered = [x for x in streak if x > 1]
            if filtered:
                non_one_streaks = len(filtered)
                avg_val = round(sum(filtered)/len(filtered), 2)
                std_dev = round(statistics.stdev(filtered), 2) if len(filtered) > 1 else 0
                median_val = statistics.median(filtered)
                max_value = max(filtered)
            else:
                non_one_streaks = 0
                avg_val = 0
                std_dev = 0
                median_val = 0
                max_value = 0
            
            f.write(
                f"{f'{v1} - {edge} - {v2}:':<46} {count:>5} "
                f"({percentage:5.2f}%); Streak Info:  "
                f"Streak_Num: {non_one_streaks:>3}  "
                f"Avg: {avg_val:>7}  "
                f"Std_Dev: {std_dev:>7}  "
                f"Median: {median_val:>7}  "
                f"Max: {max_value:>7}\n")
        
        f.write("\nTop 20 highest average streaks edge triplet combinations (only streaks > 1 are considered):\n")
        filtered_combs = []
        for edge_set, streak in total_results["triplet_comb_streaks"].items():
            filtered = [x for x in streak if x > 1]
            if filtered:
                filtered_combs.append((edge_set, filtered))
        filtered_combs = sorted(filtered_combs, key=lambda t: statistics.mean(t[1]), reverse=True)[:20]
        for edge_set, filtered in filtered_combs:
            non_one_streaks = len(filtered)
            avg_val = round(sum(filtered)/len(filtered), 2)
            std_dev = round(statistics.stdev(filtered), 2) if len(filtered) > 1 else 0
            median_val = statistics.median(filtered)
            max_value = max(filtered)
            
            f.write("\n")
            for (v1, edge, v2) in edge_set:
                f.write(f"  {v1} - {edge} - {v2}\n")
            
            f.write(f"  Count: {total_results['triplet_comb_counts'][edge_set]}; Streak Info: "
                    f"Streak_Num: {len(filtered)}\tAvg: {avg_val}\tStd_Dev: {std_dev}\tMedian: {median_val}\tMax: {max_value}\n")

def generate_file_plots(name, results, output_dir):
    """
    For a given file (non-total analysis) create a single image with:
      - Top row: Bar chart (frames with x edges) and Pie chart (edge type distribution)
      - Bottom row: A wide dot plot (with error bars, individual streak points, and annotations)
        showing the top 10 highest average streak triplets (only using streaks > 1).
    """
    # Use a gridspec with a taller bottom row
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_pie = fig.add_subplot(gs[0, 1])
    ax_dot = fig.add_subplot(gs[1, :])
    
    # Bar chart: frames with x edges
    edge_counts = results["edge_counts"]
    keys = sorted(edge_counts.keys())
    counts = [edge_counts[k] for k in keys]
    ax_bar.bar(keys, counts, color='skyblue')
    ax_bar.set_title("Frames with x edges")
    ax_bar.set_xlabel("Number of edges per frame")
    ax_bar.set_ylabel("Frame count")
    
    # Pie chart: edge type distribution
    edge_type_counts = results["edge_type_counts"]
    labels = list(edge_type_counts.keys())
    sizes = list(edge_type_counts.values())
    ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax_pie.set_title("Edge Type Distribution")
    
    # Dot plot: top 10 highest average streak triplets (only streaks > 1)
    triplet_streaks = results["triplet_streaks"]
    valid_triplets = []
    for triplet, streaks in triplet_streaks.items():
        filtered = [x for x in streaks if x > 1]
        if filtered:
            valid_triplets.append((triplet, filtered))
    if valid_triplets:
        valid_triplets = sorted(valid_triplets, key=lambda x: statistics.mean(x[1]), reverse=True)[:10]
        avg_streaks = [statistics.mean(s) for _, s in valid_triplets]
        std_streaks = [statistics.stdev(s) if len(s) > 1 else 0 for _, s in valid_triplets]
        triplet_labels = [f"{v1}\n{edge}\n{v2}" for (v1, edge, v2), _ in valid_triplets]
        # Increase spacing by multiplying y positions
        y_pos = np.arange(len(valid_triplets)) * 1.5
        
        ax_dot.errorbar(avg_streaks, y_pos, xerr=std_streaks, fmt='o', color='darkred',
                        ecolor='black', capsize=5, zorder=3)
        for i, (_, streaks) in enumerate(valid_triplets):
            jitter = np.random.uniform(-0.1, 0.1, size=len(streaks))
            y_jitter = y_pos[i] + jitter
            ax_dot.scatter(streaks, y_jitter, color='blue', alpha=0.6, s=30, zorder=2)
            median_val = statistics.median(streaks)
            count_val = len(streaks)
            x_max = max(max(streaks), avg_streaks[i] + std_streaks[i])
            ax_dot.text(x_max + 0.5, y_pos[i], f"Median: {median_val} | N: {count_val}",
                        va='center', fontsize=8, color='green')
        ax_dot.set_yticks(y_pos)
        ax_dot.set_yticklabels(triplet_labels)
        ax_dot.set_xlabel("Streak Length")
        ax_dot.set_title("Top 10 Highest Avg Streak Triplets (streaks > 1)")
    else:
        ax_dot.text(0.5, 0.5, "No triplet streak data", ha='center', va='center')
        ax_dot.set_title("Top 10 Highest Avg Streak Triplets")
    
    plt.suptitle(f"Analysis Plots: {name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, f"{name}_plots.png")
    plt.savefig(output_path)
    plt.close(fig)

def generate_total_bar_pie(results, output_dir):
    """
    For total analysis, generate a figure (Total_BarPie.png) showing the bar chart and pie chart.
    """
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar chart: frames with x edges
    edge_counts = results["edge_counts"]
    keys = sorted(edge_counts.keys())
    counts = [edge_counts[k] for k in keys]
    ax_bar.bar(keys, counts, color='skyblue')
    ax_bar.set_title("Frames with x edges")
    ax_bar.set_xlabel("Number of edges per frame")
    ax_bar.set_ylabel("Frame count")
    
    # Pie chart: edge type distribution
    edge_type_counts = results["edge_type_counts"]
    labels = list(edge_type_counts.keys())
    sizes = list(edge_type_counts.values())
    ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax_pie.set_title("Edge Type Distribution")
    
    plt.suptitle("Total Analysis: Bar & Pie Charts")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, "Total_BarPie.png")
    plt.savefig(output_path)
    plt.close(fig)

def generate_total_dot_all(results, output_dir):
    """
    For total analysis, generate a dot plot (Total_Dot_All.png) with error bars
    for every edge triplet (only using streaks > 1).
    """
    triplet_streaks = results["triplet_streaks"]
    valid_triplets = []
    for triplet, streaks in triplet_streaks.items():
        filtered = [x for x in streaks if x > 1]
        if filtered:
            valid_triplets.append((triplet, filtered))
    if not valid_triplets:
        return
    # Sort by average streak descending
    valid_triplets = sorted(valid_triplets, key=lambda x: statistics.mean(x[1]), reverse=True)
    n = len(valid_triplets)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.75 * n)))
    avg_streaks = [statistics.mean(s) for _, s in valid_triplets]
    std_streaks = [statistics.stdev(s) if len(s) > 1 else 0 for _, s in valid_triplets]
    triplet_labels = [f"{v1}\n{edge}\n{v2}" for (v1, edge, v2), _ in valid_triplets]
    y_pos = np.arange(len(valid_triplets)) * 1.5
    
    ax.errorbar(avg_streaks, y_pos, xerr=std_streaks, fmt='o', color='darkred',
                ecolor='black', capsize=5, zorder=3)
    for i, (_, streaks) in enumerate(valid_triplets):
        jitter = np.random.uniform(-0.1, 0.1, size=len(streaks))
        y_jitter = y_pos[i] + jitter
        ax.scatter(streaks, y_jitter, color='blue', alpha=0.6, s=30, zorder=2)
        median_val = statistics.median(streaks)
        count_val = len(streaks)
        x_max = max(max(streaks), avg_streaks[i] + std_streaks[i])
        ax.text(x_max + 0.5, y_pos[i], f"Median: {median_val} | N: {count_val}",
                va='center', fontsize=8, color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(triplet_labels)
    ax.set_xlabel("Streak Length")
    ax.set_title("Dot Plot for Every Edge Triplet (streaks > 1)")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "Total_Dot_All.png")
    plt.savefig(output_path)
    plt.close(fig)

def generate_total_dot_comb_top20(results, output_dir):
    """
    For total analysis, generate a dot plot (Total_Dot_CombTop20.png) with error bars
    for the top 20 average streaks for combinations of edge triplets (only using streaks > 1).
    """
    comb_streaks = results["triplet_comb_streaks"]
    valid_combs = []
    for edge_set, streak in comb_streaks.items():
        filtered = [x for x in streak if x > 1]
        if filtered:
            valid_combs.append((edge_set, filtered))
    if not valid_combs:
        return
    valid_combs = sorted(valid_combs, key=lambda x: statistics.mean(x[1]), reverse=True)[:20]
    n = len(valid_combs)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.75 * n)))
    avg_streaks = [statistics.mean(s) for _, s in valid_combs]
    std_streaks = [statistics.stdev(s) if len(s) > 1 else 0 for _, s in valid_combs]
    comb_labels = []
    for (edge_set, _) in valid_combs:
        label = "\n".join([f"{v1}-{edge}-{v2}" for (v1, edge, v2) in edge_set])
        comb_labels.append(label)
    y_pos = np.arange(len(valid_combs)) * 1.5
    
    ax.errorbar(avg_streaks, y_pos, xerr=std_streaks, fmt='o', color='darkred',
                ecolor='black', capsize=5, zorder=3)
    for i, (_, streaks) in enumerate(valid_combs):
        jitter = np.random.uniform(-0.1, 0.1, size=len(streaks))
        y_jitter = y_pos[i] + jitter
        ax.scatter(streaks, y_jitter, color='blue', alpha=0.6, s=30, zorder=2)
        median_val = statistics.median(streaks)
        count_val = len(streaks)
        x_max = max(max(streaks), avg_streaks[i] + std_streaks[i])
        ax.text(x_max + 0.5, y_pos[i], f"Median: {median_val} | N: {count_val}",
                va='center', fontsize=8, color='green')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comb_labels)
    ax.set_xlabel("Streak Length")
    ax.set_title("Top 20 Highest Avg Streak Triplet Combinations (streaks > 1)")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "Total_Dot_CombTop20.png")
    plt.savefig(output_path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Analyze semantic relations in JSON files.")
    parser.add_argument("directory", type=str, help="Path to the directory containing JSON files.")
    parser.add_argument("--plots", action="store_true", help="Generate plots and save images.")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "graphs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.txt")
    
    file_results, total_results = analyze_semantic_relations(args.directory)
    write_analysis_to_file(file_results, total_results, output_file)
    
    if args.plots:
        # Generate plots for each individual file
        for filename, results in file_results.items():
            generate_file_plots(filename, results, output_dir)
        # Generate total analysis plots (three separate images)
        generate_total_bar_pie(total_results, output_dir)
        generate_total_dot_all(total_results, output_dir)
        generate_total_dot_comb_top20(total_results, output_dir)

if __name__ == "__main__":
    main()
