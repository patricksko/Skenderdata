# Synthetic Dataset Setup for Object Recognition with BlenderProc

This repository provides a setup guide for preparing synthetic datasets using [BlenderProc](https://github.com/DLR-RM/BlenderProc) and several object datasets from the BOP benchmark. The goal is to render scenes with **target objects** and **disturbance objects** for object recognition and pose estimation tasks.

## 📁 Folder Structure

Create the following subfolders:

```bash
cd resources
mkdir cc_textures
mkdir Object_Data
```

- `cc_textures/` – (Optional) Contains the CC0 texture assets used by BlenderProc for realistic rendering.
- `Object_Data/` – Stores all downloaded object models (YCB-V, T-LESS, TYO-L, HB).

## 🔽 Downloading CC0 Textures (Optional but Recommended)

To improve the realism of your synthetic scenes, you can download the **CC0 textures** using BlenderProc’s download utility:

```bash
blenderproc download cc_textures ./cc_textures/
```

This step is optional because we use own pictures for texture for better training.

## 🎯 Downloading Object Datasets

BlenderProc supports multiple datasets from the [BOP benchmark](https://bop.felk.cvut.cz/datasets/). You will need to download the following datasets:

### Target Objects
- **YCB-V** 
  This dataset contains the target objects of interest for recognition and pose estimation.

### Disturbance Objects
Download any or all of the following to use as distractors in your scenes:

- **T-LESS**
- **TUD-L (TYO-L)**  
  (Sometimes referred to as TYO-L.)
- **HB (HomebrewedDB)**

Extract all the downloaded datasets into the `resources/Object_Data/` folder. The resulting structure should look like this:

```
resources/
├── cc_textures/          # Optional textures
├── Object_Data/
│   ├── ycbv/
│   ├── tless/
│   ├── tudlight/
│   └── hb/
```

> 🔗 Make sure each folder contains the standard BOP dataset structure (i.e., `models`, `models_eval`, etc.).

## ✅ Next Steps

Once the resources are prepared, you can start creating your BlenderProc scenes by referencing the dataset paths in your configuration files or rendering scripts.

For an example rendering pipeline or to get started with scene generation, see the `examples/` folder (if provided).

## 📌 Notes

- Ensure you are using a compatible version of Blender and BlenderProc.
- Make sure all paths in your config files correctly point to the downloaded datasets.
- If you are unsure about dataset structure, refer to the [BOP Dataset Format](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md).
