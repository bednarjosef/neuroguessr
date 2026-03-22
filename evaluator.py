import torch
import numpy as np

from dataset import create_streetview_dataloader


class Evaluator:
    def __init__(self, CONFIG, classifier, transform, val_dir):
        self.device = CONFIG['device']
        self.loader = create_streetview_dataloader(CONFIG, classifier, val_dir, 'val', transform, workers=12)
        self.centers_gpu = torch.tensor(classifier.CLASS_CENTERS_XYZ, device=self.device, dtype=torch.float32)

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def calculate_consensus_distance(self, top_k_indices, true_lats, true_lons):
        # averages 3d coords of top K clusters

        # get 3D coordinates of top K centers
        centers = self.centers_gpu[top_k_indices]  # [Batch, K, 3]
        
        # get mean
        mean_center = torch.mean(centers, dim=1)  # [Batch, 3]
        
        # normalize to unit sphere
        # mean of 2 points is inside of earth so project
        mean_center = torch.nn.functional.normalize(mean_center, p=2, dim=1)
        
        # convert to lat/lon
        z, y, x = mean_center[:, 2], mean_center[:, 1], mean_center[:, 0]
        pred_lats = torch.rad2deg(torch.asin(z)).cpu().numpy()
        pred_lons = torch.rad2deg(torch.atan2(y, x)).cpu().numpy()
        
        # calc dist
        return self.haversine(true_lats.numpy(), true_lons.numpy(), pred_lats, pred_lons)

    def run(self, model, seen_clusters=None):
        model.eval()

        dists_top1 = []
        dists_top5_cons = []
        dists_top10_cons = []
        
        # accuracy
        stats = {k: {'correct': 0, 'total': 0} for k in ['top1', 'top5', 'top10']}  # , 'clim', 'land', 'soil', 'month'
        
        print('\n--- RUNNING EVALUATION ---')
        with torch.no_grad():
            for batch in self.loader:
                imgs = batch[0].to(self.device)
                lbl_loc = batch[1].to(self.device)
                true_lats = batch[2]
                true_lons = batch[3]
                # lbl_clim = batch[4].to(self.device)
                # lbl_land = batch[5].to(self.device)
                # lbl_soil = batch[6].to(self.device)
                # lbl_month = batch[7].to(self.device)
                
                if seen_clusters is not None:
                    labels_np = lbl_loc.cpu().numpy()               # [B]
                    keep_np = seen_clusters[labels_np]             # [B] bool
                    if not keep_np.any():
                        continue  # nothing in this batch is from seen clusters

                    keep_gpu = torch.from_numpy(keep_np).to(self.device)
                    imgs = imgs[keep_gpu]
                    lbl_loc = lbl_loc[keep_gpu]

                    keep_cpu = torch.from_numpy(keep_np).bool()    # CPU mask
                    true_lats = true_lats[keep_cpu]
                    true_lons = true_lons[keep_cpu]

                # forward
                out_loc = model(imgs)  # , out_clim, out_land, out_soil, out_month

                
                # --- ACCURACY METRICS ---
                
                # top-1
                _, preds_loc = torch.max(out_loc, 1)
                stats['top1']['correct'] += (preds_loc == lbl_loc).sum().item()
                stats['top1']['total'] += lbl_loc.size(0)
                
                # top-5
                _, top5 = out_loc.topk(5, 1, True, True)
                stats['top5']['correct'] += (top5 == lbl_loc.unsqueeze(1)).sum().item()
                stats['top5']['total'] += lbl_loc.size(0)

                # top-10
                _, top10 = out_loc.topk(10, 1, True, True)
                stats['top10']['correct'] += (top10 == lbl_loc.unsqueeze(1)).sum().item()
                stats['top10']['total'] += lbl_loc.size(0)

                # # Aux Tasks
                # def update_acc(name, outputs, labels):
                #     mask = labels != -1
                #     if mask.sum() > 0:
                #         _, preds = torch.max(outputs, 1)
                #         stats[name]['correct'] += (preds[mask] == labels[mask]).sum().item()
                #         stats[name]['total'] += mask.sum().item()

                # update_acc('clim', out_clim, lbl_clim)
                # update_acc('land', out_land, lbl_land)
                # update_acc('soil', out_soil, lbl_soil)
                # update_acc('month', out_month, lbl_month)

                # --- DISTANCE METRICS ---
                # top-1 distance
                pred_centers = self.centers_gpu[preds_loc]
                z, y, x = pred_centers[:, 2], pred_centers[:, 1], pred_centers[:, 0]
                pred_lats = torch.rad2deg(torch.asin(z)).cpu().numpy()
                pred_lons = torch.rad2deg(torch.atan2(y, x)).cpu().numpy()
                dists_top1.extend(self.haversine(true_lats.numpy(), true_lons.numpy(), pred_lats, pred_lons))
                
                # top-5 distance
                dists_top5_cons.extend(self.calculate_consensus_distance(top5, true_lats, true_lons))
                
                # top-10 distance
                dists_top10_cons.extend(self.calculate_consensus_distance(top10, true_lats, true_lons))

        # --- SUMMARY CALCS ---
        def calc_stats(dists, prefix):
            d = np.array(dists)
            return {
                f"{prefix}_mean_km": np.mean(d),
                f"{prefix}_median_km": np.median(d),
                f"{prefix}_geo_score": np.mean(5000 * np.exp(-d/2000))
            }

        results = {}
        # add distances
        results.update(calc_stats(dists_top1, "val/top1"))
        results.update(calc_stats(dists_top5_cons, "val/top5"))
        results.update(calc_stats(dists_top10_cons, "val/top10"))

        # add accuracy
        for k, v in stats.items():
            acc = (v['correct'] / v['total'] * 100) if v['total'] > 0 else 0
            results[f"val/acc_{k}"] = acc
            print(f"{k.upper()}: {acc:.2f}%")

        print(f"Top-1 Median: {results['val/top1_median_km']:.0f}km | Top-5 Median: {results['val/top5_median_km']:.0f}km | Top-10 Median: {results['val/top10_median_km']:.0f}km")
        
        model.train()
        return results
    