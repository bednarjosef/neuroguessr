# neuroguessr
A 430M parameter fine-tuned visual transformer that can localize any streetview image with a median distance from the actual location being 194 kilometers.
Fined-tuned [CLIP-ViT-L/14@336px](https://huggingface.co/openai/clip-vit-large-patch14-336) on 1.2 million streetview images from 300k different locations across the globe (an image in all 4 directions), parallel GPU training with torchrun (DDP) on 1x RTX 5090, and Wandb logging.

# Inspiration
This project was inspired by the well-known game [GeoGuessr](https://www.geoguessr.com/) and by the [PIGEON](https://github.com/LukasHaas/PIGEON) project. Although PIGEON does achieve overall better results in GeoGuessr than my model, it uses the entire 360° panorama during inference, therefore rendering it not very useful for real-world scenarios where only a single image or POV is available.
The model's average Geoguessr score on the validation set is 4140.

# Results
After training the model on the specified dataset (which took 2h on 1x RTX 5090) Neuroguessr achieves a respectable 194 km median distance from the actual image location (in the validation split, of course).

# Try it out
Neuroguessr is free to try out on [geo.josefbednar.com](https://geo.josefbednar.com/). Note that it was trained strictly on streetview images and therefore is best performing on images from streets, roads, with architecture or infrastructure visible. It is also trained on 512x512px images. You can freely upload high resolution images (up to a few mb) and they will be automatically scaled down for the model, however if your image is not a square it will first be scaled down such that it's smallest dimension is equal to 512px and then a square will be cropped from the center.

# Dataset
The dataset used is a great part of the success of this model. At first I was attempting to train the model on the [OSV5M dataset](https://huggingface.co/datasets/osv5m/osv5m) due to the amount of freely accessible images in it. However, multiple unsuccessfull attempts it became clear that the geological distribution of locations in this dataset is a problem. A large majority of the images are from this dataset were taken in the United States. Due to this localazation worked well on images from the US but was insufficient for images from other parts of the world (mainly not densly populated areas).
Thus, I decided to create my own dataset. [Vali](https://github.com/slashP/Vali) was used to generate 300k relatively evenly dispersed locations across the world (using the A Community World preset). I then wrote a [custom scraping tool](https://github.com/bednarjosef/streetview-scrape) to get the raw images.
