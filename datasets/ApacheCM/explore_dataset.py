import json
import sys
from collections import Counter
from datetime import datetime

def explore_dataset(input_file, output_file):
    stats = {
        "total_commits": 0,
        "authors": Counter(),
        "files_changed": Counter(),
        "extensions": Counter(),
        "commits_by_month": Counter(),
        "diff_lengths":[],
        "total_additions": 0,
        "total_deletions": 0,
        "total_diff_length": 0,
        "max_diff_length": 0,
        "total_loc": 0,
        "max_loc": 0
    }

    print("Analyzing dataset...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): 
                    continue
                
                data = json.loads(line)
                stats["total_commits"] += 1
                
                # Authors
                stats["authors"][data.get("author", "Unknown")] += 1
                
                # Dates (Time Series)
                date_str = data.get("date", "")
                if date_str:
                    try:
                        # Parse ISO 8601 to get Year-Month grouping
                        dt = datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S")
                        month_key = dt.strftime("%Y-%m")
                        stats["commits_by_month"][month_key] += 1
                    except ValueError:
                        pass
                
                # Diff length and Addition/Deletion counting
                diff = data.get("diff", "")
                diff_len = len(diff)
                stats["diff_lengths"].append(diff_len)
                stats["total_diff_length"] += diff_len
                stats["max_diff_length"] = max(stats["max_diff_length"], diff_len)
                
                for d_line in diff.split('\n'):
                    # Count actual added/removed lines (ignoring header lines like '+++' and '---')
                    if d_line.startswith('+') and not d_line.startswith('+++'):
                        stats["total_additions"] += 1
                    elif d_line.startswith('-') and not d_line.startswith('---'):
                        stats["total_deletions"] += 1
                
                # LOC stats (if present in dataset)
                loc = data.get("loc", 0)
                stats["total_loc"] += loc
                stats["max_loc"] = max(stats["max_loc"], loc)
                
                # File and Extension stats
                for file in data.get("files", []):
                    stats["files_changed"][file] += 1
                    ext = file.split('.')[-1] if '.' in file and not file.startswith('.') else 'no_extension'
                    stats["extensions"][ext] += 1

    except FileNotFoundError:
        print(f"Error: Could not find '{input_file}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON on line {stats['total_commits'] + 1}: {e}")
        sys.exit(1)

    # Group Diff Lengths into Buckets for visualization
    length_bins = { "0-1k": 0, "1k-5k": 0, "5k-10k": 0, "10k-50k": 0, "50k+": 0 }
    for length in stats["diff_lengths"]:
        if length < 1000: length_bins["0-1k"] += 1
        elif length < 5000: length_bins["1k-5k"] += 1
        elif length < 10000: length_bins["5k-10k"] += 1
        elif length < 50000: length_bins["10k-50k"] += 1
        else: length_bins["50k+"] += 1

    # Sort months chronologically
    sorted_months = sorted(stats["commits_by_month"].keys())
    time_series = {m: stats["commits_by_month"][m] for m in sorted_months}

    # Format output
    output = {
        "summary": {
            "total_commits": stats["total_commits"],
            "unique_authors": len(stats["authors"]),
            "avg_diff_length": round(stats["total_diff_length"] / max(stats["total_commits"], 1)),
            "max_diff_length": stats["max_diff_length"],
            "avg_loc": round(stats["total_loc"] / max(stats["total_commits"], 1)),
            "max_loc": stats["max_loc"]
        },
        "additions_vs_deletions": {
            "additions": stats["total_additions"],
            "deletions": stats["total_deletions"]
        },
        "length_distribution": length_bins,
        "commits_over_time": time_series,
        "top_authors":[{"name": k, "count": v} for k, v in stats["authors"].most_common(5)],
        "top_files":[{"name": k, "count": v} for k, v in stats["files_changed"].most_common(5)],
        "top_extensions": [{"name": k, "count": v} for k, v in stats["extensions"].most_common(5)]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4)
        
    print(f"Successfully processed {stats['total_commits']} commits.")
    print(f"Exploration statistics saved to -> {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_dataset.py <your_file.jsonl>")
        sys.exit(1)
        
    input_filename = sys.argv[1]
    output_filename = "exploration.json"
    
    explore_dataset(input_filename, output_filename)