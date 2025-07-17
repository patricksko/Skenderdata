# tools

## JSON2YOLO

Adapted from: [ultralytics/JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) and Perter Hönigs json2yolo repo

- Use `general_json2yolo_bop.py` for converting rendered images from blenderproc into yolo format. 

### Define those (e.g.):

```
bop_path = '/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc/bop_data'
dataset = 'Legoblock'
split_type = 'train_pbr'
whatsplit = "val" #"train"
#scenes = list(range(0,27))
scenes = [27, 28, 29]
```
To make a good train/validation split run this code twice: The first time set "whatsplit" to "train and uncomment the first "scenes" line. The second run set "wahtsplit" to "val" and uncomment the second scenes line. Check first how many scenes you rendered. In my case I rendered 29 scenes and I used the first 26 scenes for training and the last 3 scenes for validation.
and watch the magic happen!
