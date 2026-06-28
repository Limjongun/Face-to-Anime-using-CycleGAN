# Face to Anime using CycleGAN

This repository contains an implementation for transforming human faces into anime-style characters using the CycleGAN architecture. 

## Overview

The core of this project utilizes CycleGAN to perform unpaired image-to-image translation, specifically mapping features from the domain of real human faces to anime faces. 

To provide a seamless and interactive user experience, the inference pipeline is integrated with a GUI built using **[Flet](https://flet.dev/)**. This allows users to easily load an image and generate its anime counterpart through an intuitive interface.

## Features

- **CycleGAN Architecture**: Robust model for style transfer without needing paired datasets.
- **Flet UI Inference Pipeline**: A simple and clean graphical user interface that handles image inputs, runs the CycleGAN inference, and displays the transformed anime image instantly.

## Getting Started

1. Install the required dependencies (refer to the provided `environment.yml`).
2. Run the Flet inference script (e.g., `infflet.py` or similar in the `inf/` folder) to launch the GUI and begin generating anime faces.
