# EECS836 Machine Learning Project: Surface Detection in Echograms

This repository contains the complete code and data pipeline for the Spring 2025 EECS836 course project at the University of Kansas. The project focuses on detecting the **air–snow boundary** in radar echograms using a combination of weakly supervised learning and fine-tuned CNN models.

## 🔍 Overview

Radar echograms collected by CReSIS provide vertical profiles of snow and ice layers in polar regions. Manually identifying surface boundaries is tedious and does not scale across large datasets.

This project leverages:
- Weakly labeled data (via edge detection)
- Manual boundary clicks on a small subset
- A U-Net-based segmentation model
- BCE + Dice loss for improved mask quality

## 📁 Folder Structure

Project/ ├── data/ │ ├── raw/ # Original cropped + resized echograms (from CReSIS) │ ├── processed/ # Preprocessed .npy files for training │ ├── labels_manual/ # Hand-clicked labels for air-snow surface │ ├── labels_weak/ # Auto-generated weak labels using Sobel + Canny │ └── heldout_eval/ # Held-out test images (not used in training) │ ├── results/ │ ├── overlays/ # Epoch-wise overlay predictions │ ├── heldout_eval/ # Model outputs on held-out data │ └── figs/ # Summary plots and visuals for report │ ├── src/ │ ├── preprocess.py │ ├── download_data.py │ ├── manual_label_surface.py │ ├── generate_weak_labels.py │ ├── visualize_echograms.py │ ├── train.py │ ├── eval_metrics.py │ └── evaluate_heldout.py │ └── report/ └── report.tex # Final LaTeX writeup
