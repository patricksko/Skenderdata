import contextlib
import json

import cv2
import pandas as pd
from PIL import Image
from collections import defaultdict

from utils import *

def filter_images_by_im_ids(images_data, im_ids):
    filtered_images = {}
    for image_id, image_info in images_data.items():
        if int(image_id) in im_ids:
            filtered_images[int(image_id)] = image_info
    return filtered_images


def convert_coco_json(split_path='../coco/annotations/', use_segments=False, cls91to80=False, save_dir="new_dir", scene="000000", targets=None):
    coco80 = coco91_to_coco80_class()
    im_ids = [entry['im_id'] for entry in targets]

    json_dir = split_path + "/" + str('{:06d}'.format(scene)) + "/scene_gt_coco.json"

    for json_file in sorted([Path(json_dir).resolve()]):

        fn = Path(save_dir) / 'labels'  # folder name

        with open(json_file) as f:
            data = json.load(f)

        images = {str(x['id']): x for x in data['images']}
        filtered_images = filter_images_by_im_ids(images, im_ids)
        print(filtered_images)

        imgToAnns = defaultdict(list)
        # for ann in data['annotations']:
        #     imgToAnns[ann['image_id']].append(ann)

        for ann in data['annotations']:
            print(ann['image_id'])
            if ann['image_id'] in filtered_images:
                imgToAnns[ann['image_id']].append(ann)

        print(imgToAnns)

        for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
            print(img_id)

            img = images['%g' % img_id]
            h, w, f = img['height'], img['width'], img['file_name']

            bboxes = []
            segments = []
            for ann in anns:
                if ann['iscrowd']:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann['category_id'] - 1] if cls91to80 else ann['category_id'] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

                # Segments
                if use_segments:
                    if len(ann['segmentation']) > 1:
                        s = merge_multi_segment(ann['segmentation'])
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write
            print(split_path + "/" + str('{:06d}'.format(scene)) + '/' + f)
            print(str(save_dir) + "/images" + "/" + str('{:06d}'.format(scene)) + '_' + f[4:])
            print()
            shutil.copy(split_path + "/" + str('{:06d}'.format(scene)) + '/' + f, str(save_dir) + "/images" + "/" + str('{:06d}'.format(scene)) + '_' + f[4:])

            f = str('{:06d}'.format(scene)) + '_' + f[4:]
            #print(f)
            with open((fn / f).with_suffix('.txt'), 'a') as file:
                for i in range(len(bboxes)):
                    line = *(segments[i] if use_segments else bboxes[i]),  # cls, box or segments
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')


def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance. 
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all 
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...], 
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def delete_dsstore(path='../datasets'):
    # Delete apple .DS_store files
    from pathlib import Path
    files = list(Path(path).rglob('.DS_store'))
    print(files)
    for f in files:
        f.unlink()


if __name__ == '__main__':
    save_dir = make_dirs()  # output directory
    bop_path = '/home/hoenig/BOP/gdrnpp_bop2022/datasets/BOP_DATASETS'
    dataset = 'tless_random_texture'
    split_type = 'test_primesense'
    targets_file = 'test_targets_bop19.json'

    #scenes = list(range(49,60))
    scenes = list(range(1,21))
    #scenes = [2]
    print(scenes)

    with open(bop_path + "/" + dataset + "/" + targets_file) as f:
        target_data = json.load(f)

    #print(target_data)

    for scene in scenes:
        print("BOP PATH: \t\t", bop_path)
        print("DATASET: \t\t", dataset)
        print("SPLIT: \t\t\t", split_type)
        print("")
        print("File: ", bop_path + "/" + dataset + "/" + split_type + "/" + str('{:06d}'.format(scene)) + "/scene_gt_coco.json")
        split_path = bop_path + "/" + dataset + "/" + split_type

        filtered_data = [instance for instance in target_data if instance['scene_id'] == scene]
        convert_coco_json(split_path=split_path,
                            use_segments=True,
                            cls91to80=False,
                            save_dir=save_dir,
                            scene = scene,
                            targets = filtered_data)

    # zip results
    # os.system('zip -r ../coco.zip ../coco')