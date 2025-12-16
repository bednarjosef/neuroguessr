import torch, os
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import defaultdict

from h3_classification import H3Classifier
from model_clip import CLIPModel
from dataset import create_streetview_dataloader

class Refiner():
    def __init__(self, model, dataloader, device='cpu', dir='embeddings'):
        self.model = model
        self.dataloader = dataloader
        self.transform = model.eval_transform
        self.device = device
        self.dir = dir
        self.model.to(device)
        self.embeddings = {}
        self.coords = {}

        os.makedirs(dir, exist_ok=True)

        self.get_embeddings()

    def get_top_k_classes(self, top_k, image: Image.Image):
        self.model.eval()
        image = image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
            _top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
            return top_indices[0].cpu().tolist()


    def guess(self, classifier, top_k, image: Image.Image):
        """
        The Hybrid Method:
        1. Predict Top-K broad regions (H3 cells).
        2. Retrieve ALL training images inside those regions.
        3. Find the single most similar training image (Cosine Similarity).
        4. Return the lat/lon of that specific image.
        """
        # 1. Get the Query Feature (1, D)
        
        self.model.eval()
        image_input = image.convert("RGB")
        image_tensor = self.transform(image_input).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            query_feature = self.get_normalized_features(image_tensor)
            
        # 2. Get Coarse Candidates (List of Ints)
        # We reuse the logic but skip re-running the model to save time
        # (Optimization: You could run the model once and get both logits and features if you modify get_top_k)
        # For now, we just call the method:
        print(self.coords)
        print(len(self.coords))
        top_indices = self.get_top_k_classes(top_k, image_input)
        locs = [classifier.class_to_latlon(cl) for cl in top_indices]
        print(locs)

        # 3. Gather Candidates from Index
        candidate_features_list = []
        candidate_coords_list = []
        
        for class_id in top_indices:
            if class_id in self.embeddings:
                # self.embeddings[class_id] is a Tensor of shape (N_samples, Dim)
                candidate_features_list.append(self.embeddings[class_id])
                candidate_coords_list.append(self.coords[class_id])
        
        # Edge Case: If valid classes have no images (rare but possible)
        if not candidate_features_list:
            print("Warning: No images found in predicted classes. Returning (0,0).")
            return 0.0, 0.0

        # 4. Stack all candidates into one big tensor on the GPU
        # Shape: (Total_Candidates, Dim)
        all_candidates = torch.cat(candidate_features_list).to(self.device)
        all_coords = torch.cat(candidate_coords_list).to(self.device)
        
        # Ensure data types match (query might be float32, index might be float16)
        all_candidates = all_candidates.to(dtype=query_feature.dtype)

        # 5. Cosine Similarity Search
        # Matrix Multiplication: (1, Dim) @ (Dim, Total_Candidates) -> (1, Total_Candidates)
        # This calculates the dot product between the query and EVERY candidate instantly
        sims = torch.mm(query_feature, all_candidates.t())
        
        # 6. Find the winner
        best_idx = torch.argmax(sims)
        best_coord = all_coords[best_idx]
        
        # Return as (lat, lon) float tuple
        return best_coord[0].item(), best_coord[1].item()


    def get_features(self, images):
        return self.model.vision_encoder(images)
    
    def get_normalized_features(self, images):
        features = self.get_features(images)
        return F.normalize(features, p=2, dim=1)
    
    def get_embeddings(self):
        filename = 'embeddings-861-class-unfrozen-h3-val.pt'
        try:
            self.load_embeddings(filename)
        except FileNotFoundError:
            self.build_embeddings()
            self.save_embeddings(filename)
    
    def load_embeddings(self, filename):
        path = os.path.join(self.dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'File {path} does not exist.')
        
        print(f'Loading embeddings from disk...')
        data = torch.load(path, map_location=self.device)
        self.embeddings = data['embeddings']
        self.coords = data['coords']
    
    def build_embeddings(self):
        print(f'Creating embeddings for dataset...')

        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

        class_embeddings = defaultdict(list)
        class_coords = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for batch_index, batch in enumerate(self.dataloader):
                print(f'batch {batch_index+1}')
                if (batch_index+1) % 100 == 0:
                    print(f'Processing batch {batch_index+1}/{1_200_000 // CONFIG['batch_size']}')
                images = batch[0].to(self.device)
                image_classes = batch[1].to(self.device)
                image_lats = batch[2].to(self.device)
                image_lons = batch[3].to(self.device)

                features = self.get_normalized_features(images)
                features = features.cpu().half()

                for image_index, class_id in enumerate(image_classes):
                    image_coords = torch.tensor([image_lats[image_index], image_lons[image_index]], dtype=torch.float16)
                    class_embeddings[class_id].append(features[image_index])
                    class_coords[class_id].append(image_coords)

        # Convert to tensors per class
        for class_id in class_embeddings:
            self.embeddings[class_id] = torch.stack(class_embeddings[class_id])
            self.coords[class_id] = torch.stack(class_coords[class_id])

        del class_embeddings
        del class_coords

    def save_embeddings(self, filename):
        path = os.path.join(self.dir, filename)
        print(f'Saving computed embeddings to {path}...')
        state = {
            'embeddings': self.embeddings,
            'coords': self.coords
        }
        torch.save(state, path)
    

def load_model(CONFIG, ckpt_path):
    model = CLIPModel(CONFIG).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw_state = checkpoint["state_dict"]
    else:
        raw_state = checkpoint

    fixed_state = {}
    for k, v in raw_state.items():
        new_k = k
        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]

        if hasattr(model, 'model') and not new_k.startswith('model.'):
             pass 
        elif not hasattr(model, 'model') and new_k.startswith('model.'):
             new_k = new_k[len("model."):]

        fixed_state[new_k] = v
    try:
        model.load_state_dict(fixed_state, strict=True)
        print("Success: All weights (including ViT) loaded strictly.")
    except RuntimeError as e:
        print(f"Strict loading failed. Missing/Unexpected keys:\n{e}")
        res = model.load_state_dict(fixed_state, strict=False)
        print("Loaded with strict=False. Missing keys (weights not updated):", res.missing_keys)

    return model


if __name__ == '__main__':
    device = 'cpu'
    ckpt_path = 'models/neuroguessr-861-large-acw-streetview-h3-unfrozen-2-best.pth'
    dataset = 'josefbednar/streetview-acw-300k'
    countries = [
        'AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG', 'KH', 'CA', 'CL', 'CO', 'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI', 'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS', 'IN', 'ID', 'IE', 'IL', 'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY', 'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA', 'PE', 'PH', 'PL', 'PT', 'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE', 'CH', 'TW', 'TH', 'TR', 'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN',
    ]

    CONFIG = {
        'device': device,
        'model': 'ViT-L/14@336px',
        'countries': countries,
        'batch_size': 512,
        'classes': 861,
        'backbone_unfrozen': False,
        'h3_resolution': 2,
        'h3_mappings': 'h3_utils/h3_to_class_res2_min200_ring20.json',
        'h3_counts': 'h3_utils/h3_counts_res2.json',
    }

    model = load_model(CONFIG, ckpt_path)
    torch.compile(model)
    classifier = H3Classifier(CONFIG)
    dataloader = create_streetview_dataloader(CONFIG, classifier, dataset, 'train', model.eval_transform, workers=12)

    refiner = Refiner(model, dataloader, device)

    # query_img = Image.open("img.png")
    # lat, lon = refiner.guess(classifier, top_k=5, image=query_img)

    # print(f"Predicted Location: {lat}, {lon}")
