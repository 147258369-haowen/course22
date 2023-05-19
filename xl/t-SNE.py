import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from fastai.vision.all import *
from time import sleep
import os
from multiprocessing import freeze_support

def main():
    path = Path('animals')

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(path)

    # Load the trained model
    learn = load_learner('model_10_anaimals.pkl')

    # Get predictions for the validation dataset
    preds, targets = learn.get_preds(dl=learn.dls.valid)

    # Create a DataFrame for actual labels and image paths
    valid_data = pd.DataFrame.from_dict({'Actual': targets, 'Images': dls.valid.items})

    # Extract features from the penultimate layer of the model
    valid_tensors = [dls.train_ds[i][0] for i in dls.valid.get_idxs()]
    valid_processed_tensors = [learn.dls.after_item(t) for t in valid_tensors]
    valid_normalized_tensors = [learn.dls.after_batch(t.unsqueeze(0)).squeeze() for t in valid_processed_tensors]
    valid_stacked_tensors = torch.stack(valid_normalized_tensors)
    features = learn.model[0:2](valid_stacked_tensors)

    # Flatten the features and convert to a numpy array
    features_2d = features.view(features.shape[0], -1).detach().numpy()

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_2d)
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features_2d)

    # Add t-SNE columns to the DataFrame
    valid_data['t-SNE 1'] = tsne_results[:, 0]
    valid_data['t-SNE 2'] = tsne_results[:, 1]

    # Plot t-SNE
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="t-SNE 1", y="t-SNE 2",
        hue="Actual",
        palette=sns.color_palette("hls", len(valid_data['Actual'].unique())),
        data=valid_data,
        legend="full",
        alpha=0.8
    )
    plt.show()

    # Plot confusion matrix
    conf_mat = confusion_matrix(targets, preds.argmax(dim=1))
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(conf_mat, annot=True, cmap="YlGnBu", ax=ax, fmt='d', cbar=False)
    ax.set_xticklabels(dls.vocab, rotation=45)
    ax.set_yticklabels(dls.vocab, rotation=45)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()
