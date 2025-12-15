import json
from statistics import mean, median

COUNTS_PATH = "h3_utils/h3_counts_res2.json"  # change if your file is named differently

# Simple percentile function (no numpy needed)
def percentile(sorted_list, p):
    """
    p in [0, 100]
    """
    if not sorted_list:
        return None
    if p <= 0:
        return sorted_list[0]
    if p >= 100:
        return sorted_list[-1]
    k = (len(sorted_list) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_list) - 1)
    return sorted_list[f] + (sorted_list[c] - sorted_list[f]) * (k - f)

def main():
    with open(COUNTS_PATH) as f:
        counts_dict = json.load(f)

    counts = list(counts_dict.values())
    counts = [int(c) for c in counts]  # just in case they are strings

    n_cells = len(counts)
    total_images = sum(counts)
    counts_sorted = sorted(counts)

    print("=== Basic stats ===")
    print(f"Non-empty H3 cells: {n_cells}")
    print(f"Total images      : {total_images}")
    print(f"Min per cell      : {counts_sorted[0]}")
    print(f"Max per cell      : {counts_sorted[-1]}")
    print(f"Mean per cell     : {mean(counts_sorted):.2f}")
    print(f"Median per cell   : {median(counts_sorted):.2f}")
    print()

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("=== Percentiles (images per cell) ===")
    for p in percentiles:
        print(f"{p:>3}th percentile: {percentile(counts_sorted, p):.1f}")
    print()

    # Threshold table
    THRESHOLDS = [1, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000, 5000]
    print("=== Cells above thresholds ===")
    print("thr | #cells>=thr | %cells | #images in those cells | %images")
    print("----+------------+--------+------------------------+---------")
    for thr in THRESHOLDS:
        cells_ge = sum(1 for c in counts_sorted if c >= thr)
        imgs_ge = sum(c for c in counts_sorted if c >= thr)
        pct_cells = 100 * cells_ge / n_cells
        pct_imgs = 100 * imgs_ge / total_images
        print(f"{thr:>3} | {cells_ge:>10} | {pct_cells:6.2f}% | {imgs_ge:>22} | {pct_imgs:6.2f}%")
    print()

    # Show some example cells: 10 smallest (non-zero) and 10 largest
    print("=== 10 least populated cells (count, h3_id) ===")
    # find the 10 smallest values and their cells
    smallest = sorted(counts_dict.items(), key=lambda kv: int(kv[1]))[:10]
    for cell, c in smallest:
        print(f"{c:>6}  {cell}")
    print()

    print("=== 10 most populated cells (count, h3_id) ===")
    largest = sorted(counts_dict.items(), key=lambda kv: int(kv[1]), reverse=True)[:10]
    for cell, c in largest:
        print(f"{c:>6}  {cell}")

if __name__ == "__main__":
    main()
