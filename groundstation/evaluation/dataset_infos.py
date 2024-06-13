import json
import os
import tqdm
import seaborn as sns
import pandas
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_pixel_distribution(infos):
    df = pandas.DataFrame(data=[[dataset, info['total_pixels'] - sum([info['class_wise_pixels'][i] if i in info['class_wise_pixels'].keys() else 0 for i in range(90)]), sum([info['class_wise_pixels'][i] if i in info['class_wise_pixels'].keys() else 0 for i in range(90)])] for dataset, info  in infos.items()])

    sns.set()
    df.set_index(0).T.plot.pie(legend=False, subplots=True)
    plt.show()


def visualize_sample(dataset_name, id, prefix=None, max_size=640):
    from data_loader.flight_detection_dataset import FlightDetectionDataset
    import data_loader.bounding_box_util as bb_util

    data_path = os.path.join("/data2/datasets", dataset_name)

    PATHS = {
        "train": (os.path.join(data_path, "images", "train"),
                  os.path.join(data_path, "annotations", 'instances_train.json')),
        "val": (os.path.join(data_path, "images", "val"),
                os.path.join(data_path, "annotations", 'instances_val.json')),
    }

    img_folder = PATHS['train'][0]
    ann_file = PATHS['train'][1]

    dataset = FlightDetectionDataset(img_folder, ann_file,
                                     max_whole_image_size=max_size, cache_images_in_memory=False)

    assert (0 <= id < len(dataset))

    sample = dataset[id]
    img = sample['img']
    annots = sample['annot']

    for annot in annots:
        img = bb_util.add_bbox_xyxy(img, annot[0], annot[1], annot[2], annot[3], color='orange')

    os.makedirs("/tmp/plots/ratio", exist_ok=True)
    if prefix is not None:
        plt.imsave(f"/tmp/plots/ratio/{prefix}{dataset_name}.png", img)
    else:
        plt.imsave(f"/tmp/plots/ratio/{dataset_name}.png", img)

    return img


def print_object_sizes(infos):
    for dataset, info in infos.items():
        print(f"{dataset}:")
        average_object_size = np.stack(info['object_size']).mean(0)
        average_object_size_normalized = np.stack(info['object_size_normalized']).mean(0)
        print(f"\tobject count: {info['annotations']}:")
        print(f"\tmax objects in one image: {info['max_objects_in_one_image']}")
        print(f"\taverage object size: {average_object_size}:")
        print(f"\taverage object size (normalized):{average_object_size_normalized}:")


def pixel_ratio(dataset, tiling=False):
    dataset_path = os.path.join("/data2/datasets", dataset, "annotations/instances_train.json")

    if tiling:
        from data_loader.flight_detection_dataset_splitter_v3 import FlightDetectionDataset
        d = FlightDetectionDataset(os.path.join("/data2/datasets", dataset, "images/train"), dataset_path, None)
    else:
        from data_loader.flight_detection_dataset import FlightDetectionDataset
        d = FlightDetectionDataset(os.path.join("/data2/datasets", dataset, "images/train"), dataset_path, None)

    image_pixels = 0
    object_pixels = 0
    image_ratio = []

    for sample in tqdm.tqdm(d, desc=f"Dataset: {dataset}"):
        img, annots = sample['img'], sample['annot']

        image_pixels += img.shape[0] * img.shape[1]
        image_ratio.append(img.shape[1] / img.shape[0])
        object_pixels += ((sample['annot'][:, 2] - sample['annot'][:, 0]) * (sample['annot'][:, 3] - sample['annot'][:, 1])).sum()

    print(f"Dataset {dataset} [tiling: {tiling}]")
    print(f"\t foreground: {object_pixels}")
    print(f"\t background: {image_pixels - object_pixels}")
    print(f"\t ratio of foreground: {object_pixels / image_pixels}")


def create_info(dataset):
    dataset_path = os.path.join("/data2/datasets", dataset, "annotations/instances_train.json")
    dataset_json = json.load(open(dataset_path))

    info = {
        'images': len(dataset_json['images']),
        'image_shapes': [],
        'annotations': len(dataset_json['annotations']),
        'total_pixels': 0,
        'class_wise_pixels': {},
        'categories': dataset_json['categories'],
        'object_size': [],
        'object_size_normalized': [],
        'max_objects_in_one_image': 0
    }

    anns_per_img = {}
    for ann in tqdm.tqdm(dataset_json['annotations'], desc="Collect annotations per image"):
        anns_per_img.setdefault(ann['image_id'], []).append(ann)

    for img in tqdm.tqdm(dataset_json['images'], desc="Check images"):
        img_pixels = img['width'] * img['height']
        info['image_shapes'].append((img['width'], img['height']))

        anns = anns_per_img[img['id']] if img['id'] in anns_per_img.keys() else []

        for ann in anns:
            class_id = ann['category_id']
            info['class_wise_pixels'][class_id] = info['class_wise_pixels'].setdefault(class_id, 0) + ann['bbox'][2] * \
                                                   ann['bbox'][3]
            info['object_size'].append((ann['bbox'][2], ann['bbox'][3]))
            info['object_size_normalized'].append((ann['bbox'][2] / img['width'], ann['bbox'][3] / img['height']))

        info['total_pixels'] += img_pixels

    info['max_objects_in_one_image'] = max([len(anns) for anns in anns_per_img.values()])
    return info


if __name__ == '__main__':

    # for d in ['vis_drone', 'lake_constance_v2021','coco' , 'dota_20']:
    #     pixel_ratio(d)
    #     pixel_ratio(d, tiling=True)

    #visualize_sample("vis_drone", 174, prefix='procedure_', max_size=1024)

    visualize_sample("lake_constance_v2021", 60)
    visualize_sample("vis_drone", 58)
    visualize_sample("coco", 212)
    visualize_sample("dota_20", 70)

    infos = {}
    for dataset in ['coco', 'vis_drone', 'lake_constance_v2021', 'dota_20']:
        infos[dataset] = create_info(dataset)

    print_object_sizes(infos)

#    print(infos)
    plot_pixel_distribution(infos)