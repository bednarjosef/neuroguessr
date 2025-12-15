import json
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
import h3

# =========================
# CONFIG
# =========================
COUNTS_JSON_PATH = "h3_counts_res2.json"  # path to your counts JSON
H3_RESOLUTION = 2                          # resolution used for those counts

MIN_SAMPLES = 200                          # <- your new minimum
MAX_NEIGHBOR_RING = 20                     # <- your big neighbor ring

MAPPING_OUT_PATH = f"h3_to_class_res{H3_RESOLUTION}_min{MIN_SAMPLES}_ring{MAX_NEIGHBOR_RING}.json"

# =========================
# H3 compatibility helpers (v3 & v4)
# =========================
try:
    from h3 import api
    H3_V4 = True
except ImportError:
    api = None
    H3_V4 = False

def h3_k_ring(cell, k):
    """Return neighbors within k rings, for both h3 v3 and v4."""
    if hasattr(h3, "k_ring"):  # v3
        return h3.k_ring(cell, k)
    elif H3_V4:
        return api.basic_str.grid_disk(cell, k)
    else:
        raise RuntimeError("No k_ring/grid_disk found in h3")

def h3_to_boundary(cell, geo_json=True):
    """
    Get polygon boundary for a cell.
    For v4, cell_to_boundary returns (lat, lon); we flip to (lon, lat) if geo_json=True.
    """
    if hasattr(h3, "h3_to_geo_boundary"):  # v3
        return h3.h3_to_geo_boundary(cell, geo_json=geo_json)
    elif H3_V4:
        boundary = api.basic_str.cell_to_boundary(cell)  # list of (lat, lon)
        if geo_json:
            return [(lon, lat) for lat, lon in boundary]  # (lon, lat)
        else:
            return boundary
    else:
        raise RuntimeError("No boundary function found in h3")

# =========================
# Load counts
# =========================
with open(COUNTS_JSON_PATH) as f:
    counts = json.load(f)

counts = {cell: int(c) for cell, c in counts.items()}
print(f"Non-empty H3 cells: {len(counts)}")
print(f"Total images       : {sum(counts.values())}")

# =========================
# Build merged mapping (small cells -> nearest big cell)
# =========================
big_cells = {cell for cell, c in counts.items() if c >= MIN_SAMPLES}
print(f"Big cells (≥{MIN_SAMPLES} imgs): {len(big_cells)}")

sorted_big = sorted(big_cells, key=lambda c: counts[c], reverse=True)
h3_to_class = {cell: i for i, cell in enumerate(sorted_big)}
print(f"Initial number of classes (big cells): {len(h3_to_class)}")

unassigned = 0
for cell, c in counts.items():
    if cell in big_cells:
        continue

    assigned = False
    for k in range(1, MAX_NEIGHBOR_RING + 1):
        for nb in h3_k_ring(cell, k):
            if nb in big_cells:
                h3_to_class[cell] = h3_to_class[nb]
                assigned = True
                break
        if assigned:
            break

    if not assigned:
        unassigned += 1

print(f"Small cells with NO big neighbor within {MAX_NEIGHBOR_RING} rings: {unassigned}")
print(f"Total cells with an assigned class: {len(h3_to_class)} out of {len(counts)}")

# =========================
# Save mapping for training
# =========================
with open(MAPPING_OUT_PATH, "w") as f:
    json.dump(h3_to_class, f)

print(f"Saved mapping to: {MAPPING_OUT_PATH}")

# =========================
# Build GeoDataFrame (per hex)
# =========================
def h3_to_polygon(cell):
    boundary = h3_to_boundary(cell, geo_json=True)  # list of (lon, lat)
    return Polygon(boundary)

records = []
for cell, c in counts.items():
    if cell not in h3_to_class:
        continue  # skip unassigned
    geom = h3_to_polygon(cell)
    cls = h3_to_class[cell]
    is_big = cell in big_cells
    records.append(
        {
            "h3": cell,
            "count": c,
            "class_id": cls,
            "is_big": is_big,
            "geometry": geom,
        }
    )

gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
print(f"GeoDataFrame rows (assigned cells): {len(gdf)}")

gdf["geometry"] = gdf["geometry"].buffer(0)

# =========================
# Dissolve by class_id -> one polygon/multipolygon per final class
# =========================
gdf_classes = gdf.dissolve(by="class_id")  # index is class_id
gdf_classes = gdf_classes.reset_index()
print(f"Number of final classes: {len(gdf_classes)}")

# =========================
# Plot: one outline per merged class
# =========================
fig, ax = plt.subplots(figsize=(16, 8))

# light fill per class (colors repeat, that's ok)
gdf_classes.plot(
    ax=ax,
    column="class_id",
    cmap="tab20",
    alpha=0.6,
    linewidth=0.0,
    edgecolor="none",
    categorical=True,
    legend=False,
)

# single border around each merged class
gdf_classes.boundary.plot(
    ax=ax,
    color="black",
    linewidth=0.5,
)

ax.set_title(
    f"Merged geo-classes outlines (MIN_SAMPLES={MIN_SAMPLES}, res={H3_RESOLUTION})",
    fontsize=16,
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()
