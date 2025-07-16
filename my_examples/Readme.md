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
> ⚠️ File naming may vary slightly between datasets (e.g. `camera_intrinsics.json`). Adjust as needed or unify them under the expected name (`camera.json`) for compatibility.

