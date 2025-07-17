## 📁 Dataset Setup

Before running the project, you need to prepare the datasets. Follow the steps below:

### 1. Create the `resources` Directory

In the root of the project, create a folder named:

resources/


### 2. Download BOP Datasets

Download the following datasets from the official BOP benchmark website:  
👉 [https://bop.felk.cvut.cz/datasets/](https://bop.felk.cvut.cz/datasets/)

Required datasets:

- `ycbv`
- `tless`
- `hb` (HomebrewedDB)
- `itodd`
- `tyol`

### 3. Extract Each Dataset

For each dataset, extract it inside the `resources` folder so the structure looks like this:

resources/

├── hb/

│   ├── models/

│   ├── models_eval/

│   └── camera.json

├── ycbv/

│   ├── models/

│   ├── models_eval/

│   └── camera.json

├── tless/

│   ├── models/

│   ├── models_eval/

│   └── camera.json

├── itodd/

│   ├── models/

│   ├── models_eval/

│   └── camera.json

├── tyol/

│   ├── models/

│   ├── models_eval/

│   └── camera.json




> ⚠️ **Note:** Make sure each dataset contains the following inside its root directory:
> - `models/` — 3D object models  
> - `models_eval/` — evaluation-ready models  
> - `camera.json` — camera intrinsics file  
> 
> ⚠️ File naming may vary slightly between datasets (e.g. `camera_intrinsics.json`).

## 🎨 Download CC Textures

Next, download the [CC Textures] using BlenderProc's built-in downloader.

```bash
blenderproc download cc_textures <output_dir>
```

Replace `<output_dir>` with the path where you want to save the textures.  
I recommend placing them inside the `resources/` folder like this:

```bash
resources/
├── cc_textures/
│   ├── wood/
│   ├── metal/
│   └── ...
├── Objects/
│   ├── ycbv/
│   ├── tless/
│   └── ...
```

---

## ⚙️ Configuration

Once all datasets and textures are in place, open the `config.json` file in this repository.

Update the paths inside it to point to your local folders.

---
For compatibility with Peter Hönigs JSON2YOLO file you need to do some changes in:
```bash
blender/blender-4.2.1-linux-x64/custom-python-packages/lib/python3.11/site-packages/bop_toolkit_lib/pycoco_utils.py
```

- In the "create_annotation_info" function change mask_encoding_format from "rle" to "polygon"
- In the "binary_mask_to_polygon" function change
```
contours = np.array(measure.find_contours(padded_binary_mask, 0.5))
```

to
```
contours = measure.find_contours(padded_binary_mask, 0.5)
```
and change this code line
```
contours = contours - 1
```
to 
```
contour = contour - 1
```
and put it in the for loop
You're now ready to run the rendering_tless.py script in the bop_example folder with:
```bash
blenderproc run <path/to>rendering_tless.py --config=<path/to>config.json

