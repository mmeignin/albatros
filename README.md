# albatros Smoke Generation Project

# Smoke generator V1
The Smoke Generator Project is a versatile tool for generating visually appealing composite images of smoke plumes against a variety of backgrounds. It combines Blender for smoke simulation and Python for image composition to create dynamic and captivating smoke images.

<p align="center">
  <img src="readme.png" alt="Example Smoke Composite Image" width="400"/>
</p>

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Project Overview

The Smoke Generator Project provides a simple yet effective solution for creating realistic smoke plume composite images. The project leverages Blender for smoke simulation and image composition and aims to offer flexibility in generating smoke images with varying properties, such as rotation and brightness adjustments.

## Project Structure

# Project Structure

This document outlines the structure of the Smoke Generation Project for version 1.

## Directory Structure

- **smoke_dataset_V1**: The primary folder for version 1 of the smoke dataset. Contains image and mask data.
  - 📁 images: Directory for smoke dataset images.
  - 📁 masks: Directory for corresponding masks.

- **smoke_generator_V1**: The primary folder for version 1 of the smoke generation code. Includes various subdirectories and scripts.
  - 📁 background_images: Contains background images used for composing smoke images.
  - 📁 blender_files: Contains Blender-related files and scripts for smoke simulation.
    - 🐍 blender_main.py: Orchestrates the generation of smoke simulations within Blender.
    - 🐍 blender_scene_utils.py: Utility methods for managing scene operations in Blender.
    - 💻 random_smoke_plume.blend: Template/base Blender file for creating smoke simulations.
  - 📁 blender_images: Holds numbered folders for different smoke plume simulations (e.g., 'smokeplume_0').
  - 📁 scripts: Contains various Python scripts related to the project.
    - 🐍 bb_EDA.py: Conducts exploratory data analysis (EDA) for rescaling size.
    - 🐍 display_project_architecture.py: Visualizes the project's directory structure.
    - 🐍 install_harmonization.py: Manages the image harmonization process.
    - 🐍 main.py: The main entry point script for running the project.
    - 🐍 mask_eda.py: Conducts EDA for assessing smoke mask quality.
    - 🐍 sky_eda.py: Conducts EDA for non-sky regions.
    - 🐍 test_transforms.py: Contains scripts for testing image transformations.
    - 📁 utils: Utility scripts and modules.
      - 🐍 application_interface.py: Manages user interaction and UI.
      - 🐍 image_composition.py: Composes and processes smoke-related images.
      - 🐍 image_harmonization.py: Harmonizes image qualities.
      - 📁 target_images: Target images used in transformations.
      - 🐍 transform_methods.py: Contains various image transformation methods.

- **smoke_generator_V2**: Placeholder for version 2 of the smoke generation code.
  - 📁 model: Contains scripts related to the machine learning model for smoke generation.
    - 🐍 custom_dataset.py: Custom dataset class for training the model.
    - 🐍 training.py: Script for training the smoke generation model.
    - 🐍 unet.py: Definition of the UNet architecture used in the model.
  - 📁 weight: Directory for storing model weights and checkpoints.

## Environment Setup

- **smoke_generator_env**: This directory contains the Python virtual environment used for library management.
  - 🔧 create_python_env.sh: A script to create and set up the Python virtual environment.
  - 📄 python_dependencies.txt: A list of Python package dependencies required for the project.



## Getting Started

### Requirements

- [Blender](https://www.blender.org/) (version 3.41 or later)
- [Python](https://www.python.org/) (version 3.10 or later)
- Python libraries:
  - Pillow==8.3.1
  - matplotlib==3.4.3
  - torch==1.9.1
  - torchvision==0.10.1
  - gdown==4.7.1
  - cv2
  - tk
  - scikit image
  - pandas 
  - numpy 
### Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/mmeignin/albatros.git
cd albatros
sh create_python_env.sh
#activate python env
python3 install_harmonization.py
python3 main.py
```

### Usage

1. Open Blender and run the `blender_main.py` script in the `blender_file` folder to generate smoke plume images in the `blender_images` folder.

2. Run the `main.py` to perform composition between smoke images and background images

3. Experiment with the configuration options in the `main.py` script to customize the appearance of the composite images.

## Configuration


## Acknowledgments

The Smoke Generator Project is inspired by [cite the source if applicable].


Please replace `[License Name]`, update the sections with actual links, versions, and relevant information specific to your project, and add any additional sections as needed.
