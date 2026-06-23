import os
import cv2
import numpy as np

base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(base, 'data')

labels = ['PNEUMONIA', 'NORMAL']
img_size = 224


def count_files(folder_path):
    counts = {}
    for label in labels:
        p = os.path.join(folder_path, label)
        if os.path.isdir(p):
            counts[label] = len([f for f in os.listdir(p) if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg') or f.lower().endswith('.png')])
        else:
            counts[label] = 0
    return counts


def get_data_paths(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        if not os.path.isdir(path):
            continue
        for img in os.listdir(path):
            if not (img.lower().endswith('.jpeg') or img.lower().endswith('.jpg') or img.lower().endswith('.png')):
                continue
            full = os.path.join(path, img)
            data.append((full, class_num))
    return data


def analyze():
    for split in ['train', 'val', 'test']:
        folder = os.path.join(DATA_DIR, split)
        print('\n==', split, '==')
        counts = count_files(folder)
        print('file counts by class:', counts)
        data = get_data_paths(folder)
        print('total images listed:', len(data))
        # check class distribution
        labels_arr = [c for (_, c) in data]
        unique, counts = np.unique(labels_arr, return_counts=True) if labels_arr else ([], [])
        print('unique label ids:', unique, 'counts:', counts)
        # sample pixel stats for first 5 images
        sample = data[:5]
        for path, c in sample:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print('failed to load', path)
                continue
            resized = cv2.resize(img, (img_size, img_size))
            print('sample', os.path.basename(path), 'label', c, 'shape', resized.shape, 'min/max', resized.min(), resized.max(), 'dtype', resized.dtype)

    # check for filename overlaps across splits
    paths_by_split = {s: set([os.path.basename(p) for p, _ in get_data_paths(os.path.join(DATA_DIR, s))]) for s in ['train', 'val', 'test']}
    for a in ['train', 'val', 'test']:
        for b in ['train', 'val', 'test']:
            if a >= b:
                continue
            overlap = paths_by_split[a].intersection(paths_by_split[b])
            print(f'Overlap between {a} and {b}:', len(overlap))
            if len(overlap) > 0:
                print('example overlap (up to 10):', list(overlap)[:10])

if __name__ == '__main__':
    analyze()

