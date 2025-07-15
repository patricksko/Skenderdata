# tools

## JSON2YOLO

Adapted from: [ultralytics/JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)

- Use `general_json2yolo_bop.py` if you have NO target file.
- Use `general_json2yolo_bop_targets.py` if you have a target file (e.g. for test data BOP format).

### Define those (e.g.):

```
bop_path = '/home/hoenig/BOP/Pix2Pose/pix2pose_datasets'
dataset = 'itodd'
split_type = 'val'
scenes = list(range(1,2))
```

or

```
bop_path = '/home/hoenig/BOP/Pix2Pose/pix2pose_datasets'
dataset = 'tless'
split_type = 'train_pbr'
scenes = list(range(50))
```
`python general_json2yolo_bop.py`

and watch the magic happen!