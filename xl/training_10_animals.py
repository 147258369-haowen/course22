from duckduckgo_search import ddg_images
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
from time import sleep
import os
from multiprocessing import freeze_support


def main():
 
    path = Path('animals')
    failed = verify_images(get_image_files(path))
    failed.map(Path.unlink)
    len(failed)
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))
    for i, ax in enumerate(axs.flatten()):
        xb, yb = dls.one_batch()
        img, label = xb[i], yb[i]
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(dls.vocab[label])
    plt.tight_layout()
    plt.show()

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)
    learn.export('model_10_anaimals.pkl')

    # # Replace 'bird.jpg' with a test image filename
    
    # test_image = 'loin.jpg'
    # learn = load_learner('model_10_anaimals.pkl')
    # predicted_animal, _, probs = learn.predict(PILImage.create(test_image))
    # predicted_probability = probs.max().item()

    # print(f"This is a: {predicted_animal}.")
    # print(f"Probability it's a {predicted_animal}: {predicted_probability:.4f}")

if __name__ == '__main__':
    freeze_support()
    main()