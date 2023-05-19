from duckduckgo_search import ddg_images
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
from time import sleep
import os
from multiprocessing import freeze_support
def search_images(term, max_images=200): return L(ddg_images(term, max_results=max_images)).itemgot('image')

def main():
    # searches = ['cat', 'dog', 'elephant', 'lion', 'monkey', 'penguin', 'giraffe', 'tiger', 'bear', 'zebra']
    searches = [ 'zebra']
    path = Path('animals')
    for o in searches:
        dest = (path/o)
        dest.mkdir(exist_ok=True, parents=True)
        download_images(dest, urls=search_images(f'{o} photo'))
        sleep(3)  # Pause between searches to avoid over-loading server
        download_images(dest, urls=search_images(f'{o} sun photo'))
        sleep(3)
        download_images(dest, urls=search_images(f'{o} shade photo'))
        sleep(3)
        resize_images(path/o, max_size=400, dest=path/o)
if __name__ == '__main__':
    freeze_support()
    main()