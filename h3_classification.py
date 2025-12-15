import json, math, h3

import numpy as np


class H3Classifier():
    def __init__(self, CONFIG):
        self.resolution = CONFIG['h3_resolution']

        self.load_mappings(CONFIG['h3_mappings'])
        self.load_counts(CONFIG['h3_counts'])
        self.calculate_weighted_centers()

    def class_to_latlon(self, class_id):
        x, y, z = self.CLASS_CENTERS_XYZ[class_id]
        lat = math.degrees(math.asin(z))
        lon = math.degrees(math.atan2(y, x))
        return lat, lon
    
    def latlon_to_class(self, lat, lon):
        cell = h3.latlng_to_cell(lat, lon, self.resolution)
        return self.H3_TO_CLASS.get(cell)
        
    def load_mappings(self, mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)

        self.H3_TO_CLASS = {cell: int(cls) for cell, cls in mappings.items()}
        self.NUM_CLASSES = max(self.H3_TO_CLASS.values()) + 1

    def load_counts(self, counts_file):
        with open(counts_file, 'r', encoding='utf-8') as f:
            counts = json.load(f)

        self.COUNTS = {cell: int(c) for cell, c in counts.items() if cell in self.H3_TO_CLASS}

    def calculate_weighted_centers(self):
        sum_vec = np.zeros((self.NUM_CLASSES, 3), dtype=np.float64)
        sum_w = np.zeros(self.NUM_CLASSES, dtype=np.float64)

        for cell, cls in self.H3_TO_CLASS.items():
            w = self.COUNTS.get(cell, 1)

            # center lat/lon of this H3 cell
            lat, lon = h3.cell_to_latlng(cell)
            lat_r = math.radians(float(lat))
            lon_r = math.radians(float(lon))

            # convert to unit sphere xyz
            x = math.cos(lat_r) * math.cos(lon_r)
            y = math.cos(lat_r) * math.sin(lon_r)
            z = math.sin(lat_r)

            # accumulate weighted vector
            sum_vec[cls, 0] += x * w
            sum_vec[cls, 1] += y * w
            sum_vec[cls, 2] += z * w
            sum_w[cls] += w

        centers = np.zeros_like(sum_vec)
        for i in range(self.NUM_CLASSES):
            if sum_w[i] > 0:
                v = sum_vec[i] / sum_w[i]
                norm = np.linalg.norm(v)
                if norm > 0:
                    centers[i] = v / norm
                else:
                    centers[i] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                centers[i] = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        self.CLASS_CENTERS_XYZ = centers.astype(np.float32)
