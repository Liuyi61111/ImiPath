# ImiPath

Official repository for **ImiPath**, including code, datasets, trained models, experimental results, supplementary materials, and demo videos.

## Overview

ImiPath is a learning-based path planning framework designed for decision-making under partial observability.  
This repository provides:

## Repository Structure

The main files and folders in this repository are organized as follows:

- **Dataset.zip**  
  Contains the map data used in the experiments.

- **models/**  
  Stores the network checkpoints generated at different stages of training.

- **training.log**  
  Records the training process of the model, including intermediate outputs and training progress.

- **ELO/**  
  Contains additional records of how the model's problem-solving capability evolves during training, measured with an ELO-style evaluation protocol.

- **PFACO/**  
  Provides the implementation of the latest ACO-based algorithm mentioned in the manuscript.

- **Pure_aco/**  
  Contains the baseline implementation of the conventional Ant Colony Optimization (ACO) method.

- **results/**  
  Stores the experimental results of **STAPNet** on the **FoV scale** test set, including 100 experiment records with metrics such as path length, computation time, and turning-related statistics.

- **Dynamic Environments1.mp4**, **Dynamic Environments2.mp4**, **Dynamic Environments3.mp4**  
  Experimental videos corresponding to the dynamic environment evaluation.

- **ExperimentalResultsonMagneticMicrorobotPlatform...**  
  Experimental video demonstrating the results on the magnetic microrobot platform.

- **Supplementary_ImiPath.pdf**  
  Supplementary material for the paper.

- **train_multi.py**  
  Main training script for model training.

- **random100_2.1.1.npy**  
  Contains a collection of 100 test tasks for the **FoV scale** evaluation, including randomly generated maps as well as random start and goal positions.

## Notes

- The repository includes both the proposed method and baseline algorithms for comparison.
- Experimental videos are provided to visually demonstrate the performance of the method in dynamic environments and on the physical magnetic microrobot platform.
- Additional implementation details and experimental analysis can be found in the supplementary PDF.
