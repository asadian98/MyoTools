# MyoTools

Welcome to the **MyoTools** repository! This project contains tools for **data extraction**, **motor unit sorting**, and **basic visualization** for neuroscience research. The code is designed to work seamlessly with data from **MonkeyLogic** and the **Myomatrix electrodes** recorded using **Open Ephys recording system**. 

## Overview

This repository provides:
- Scripts for extracting data from MonkeyLogic and OEPHYS.
- Spike sorting tools using popular external libraries.
- Basic analysis and visualization functions with an easy-to-use GUI.

The core pipeline is defined in **`MyoPipeline_main.m`**, with detailed comments about the outputs and file organization.

---

## Dependencies

MyoTools leverages several external repositories for motor unit sorting and analysis. These are included in the repository for convenience:

| Tool              | Description                                    | Link                                                                                         |
|-------------------|------------------------------------------------|----------------------------------------------------------------------------------------------|
| **EMUsort**           | high-performance spike sorting of multi-channel, single-unit electromyography | [EMUsort](https://github.com/snel-repo/EMUsort)                                                      |
| **emg2mu**           | GPU-Accelerated High-Density EMG Decomposition | [emg2mu](https://github.com/neuromechanist/emg2mu)                                                      |
| **Phy**           | GUI for manual curation of spike sorting       | [Phy](https://github.com/kwikteam/phy)                                                      |

---

## Getting Started

### Setup
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/asadian98/MyoTools.git

2. Open MATLAB and navigate to the repository folder.
3. Start the pipeline with:
    ```bash
    MyoPipeline_main.m
### File Organization
Output files and their organization are explained in the comments within MyoPipeline_main.m.

### License
This project is open-source and distributed under the MIT License. Please review the license file for details.
