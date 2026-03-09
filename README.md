# ROAM: Real-time Objective Animal Monitoring Toolkit 
## Associated article: Towards 24/7 behavioural monitoring: Automated real-time surveillance of animal behaviour from continuous video streams
Li-Dunn Chen, Molly McGuire, Gabriela Mastromonaco (Toronto Zoo Wildlife Health Unit)

Stephen Dodds (Juuk Inc.)

### This dataset is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
Use permitted for academic and non-commercial research purposes only.
Commercial use is prohibited without explicit permission from the authors.

## Description
This repository contains code for the ROAM behavioural monitoring framework, a computer vision method that can be used for real-time monitoring of CCTV livestreams and more for generating metrics such as activity budget, space use heatmaps, and estbalishing real-time alert systems.


![banner](./Images/GraphicalAbstract.png)


## Abstract

> 1.	Behavioural monitoring is central to understanding how animals respond to naturally occurring and human-induced environmental change, yet scalable approaches for behavioural data collection, analysis, and interpretation remain limited. Zoological institutions provide a valuable setting for developing and validating automated behavioural monitoring tools, as they encompass complex environments containing wildlife while offering a degree of control relative to conditions in the wild. 
2.	Although recent advances in machine learning (ML) facilitate behavioural inference from video data, few approaches translate these capabilities into real-time, alert-enabled monitoring systems capable of continuous deployment. We present a generalizable ML framework that integrates predictive models with CCTV livestreams to support automated animal behaviour surveillance, using polar bears (Ursus maritimus), giraffes (Giraffa reticulata), and Siberian Jays (Perisoreus infaustus) as focal species. 
3.	This framework conducts real-time classification of routine behaviours (foraging, locomotion, resting, swimming), while stereotypic behaviours (head swinging and loop swimming) are inferred using logic-based heuristics applied to frame-wise model outputs. The system supports customizable alert thresholds based on behaviour duration and/or frequency, with automated notifications delivered to designated personnel.
4.	Broad applicability of the framework was demonstrated by applying it to publicly available datasets spanning distinct taxa and study contexts, where it generated real-time behavioural alerts and automated post-monitoring summaries with modification only to the detection and behaviour-definition layers, requiring minimal adjustment to the core pipeline.
5.	Beyond real-time alerts, the framework streamlines downstream data processing and generates interpretable outputs, including activity budgets and spatiotemporal visualizations. Altogether, this integrated end-to-end pipeline provides a transferable and cost-efficient method for converting continuous video streams into structured behavioural datasets suitable for quantitative analysis across welfare science, conservation management, and applied behavioural research. 



## Quick Start
We provide the whole computer vision pipeline from data annotation to model training and deployment in the associated Ecological Informatics article linked above. Here, we will run a quick demo inference visualization. Make sure you download the [sample dataset containing images and associated behavioural labels](https://zenodo.org/records/17497020?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImUxOGFmMWFlLTI3MzUtNGMzNC1hYzJjLWJiODQ0NGI5MmQwOSIsImRhdGEiOnt9LCJyYW5kb20iOiI4ODk0OTI3MDNlZmFkMTk3NWVlMmZlZDBjMDM5ZjRkNiJ9.126GeRvizJagtLPX5JzkdT6-ZJZq6jRpwoJILhKOAopVHt_dkcuMXDEQ-s4lyzgt-FKpj_BKUvpVJc2IX12XRA). In the near future, video walkthroughs will be provided.

### Installation
There are a series of required packages to run the pipeline. We recommend creating a [conda environment](https://www.anaconda.com/). For ease of deployment, we recommend the use of a command line interface such as Anaconda Prompt or a code editor such as like VS Code. 

You can create a new environment and install required packages by running:
```
conda create -n ENVIRO python=3.8 
conda activate ENVIRO
cd Desktop/PantherAI

pip install -r requirements.txt
```
* "ENVIRO" is the name of the working environment and can use any name
* Use the cd command to set the file directory to where all your data and script are located

## PantherAI Dataset

The provided dataset contains ~14,000 labeled images across 5 behavioural classes (feeding, locomotion, object manipulation, resting, and stereotypical pacing). It includes:

- `/images/` — raw images
- `/labels/` — YOLO-format text labels per image

#### 📦 Download

**Video Frames and Associated Labels (51 GB total)**:  
[Download from Zenodo](https://zenodo.org/records/17497020?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImUxOGFmMWFlLTI3MzUtNGMzNC1hYzJjLWJiODQ0NGI5MmQwOSIsImRhdGEiOnt9LCJyYW5kb20iOiI4ODk0OTI3MDNlZmFkMTk3NWVlMmZlZDBjMDM5ZjRkNiJ9.126GeRvizJagtLPX5JzkdT6-ZJZq6jRpwoJILhKOAopVHt_dkcuMXDEQ-s4lyzgt-FKpj_BKUvpVJc2IX12XRA)

## Run Inference on Sample Data
After installation of required softare packages and downloading the sample dataset, the below series of scripts should be executable through the Anaconda Prompt environment. Ensure that the current working directory is set to `PantherAI_Repo` or the source location of your project files. You can change your working directory by using the "cd" command:  `cd /path/to/PantherAI_Repo`

Please refer to the manuscript cited below for details regarding the YOLO file structure for deploying the PantherAI pipeline. Note that several files and associated scripts are needed to 1) preprocess data, 2) prepare data for analysis, 3) train models, and 4) deploy models for real-time monitoring, as well as generating activity budget plots and space use heatmaps. 
* In addition to training data (images and associated labels/annotations), a train.txt file, .yaml file, and pretrained model weights [("yolov8m.pt"](https://huggingface.co/Ultralytics/YOLOv8)) are needed for training and deploying predictive models on livestream video footage or precorded video sequences. These files can be found in the [Data](./Data) and [Code](./Code) folders.
  
![PantherAI Scripts](./Images/Figure3.png)

```
python 10_video_inference.py
```
[![Watch the video](https://github.com/lidunnchen/PantherAI/blob/main/Images/Video1_Still.png)](https://github.com/lidunnchen/PantherAI/blob/main/Images/SupplementalVideo1_c28_loco_obman.mp4) 

Videos can be processed faster than real-time for retroactively assessing behavioural profiles. For example, the script, "07_activity_budget" can be used to evaluate the activity budget for any video duration and express the monitoring session as the proportion of time engaged in each behavioural category. 
![ActivityBudget](./Images/Figure6.png)

Once the model is reliably detecting behaviour(s) of interest, it is possible to send real-time alerts triggered by specific behaviours, such as stereotypical pacing. The triggering threshold can be easily adjusted (a single frame detection vs. two minutes of continuous pacing). As a single missed detection could prevent the email alert from triggering, we also set a parameter specifying the percent of detected frames (e.g., 80% instead of 100%) that would qualify and initiate a pacing alert.
![PantherAI Scripts](./Images/PantherAI_EmailAlert.png)

Furthermore, the spatial expression of behavioral categories can be visualized using the provided script:

```
python 08_heatmap_video.py
```
![Heatmap](./Images/Figure7_heatmap.png)
* Space usage heatmaps automatically generated from test data using PantherAI. The prevalence of each behaviour: A) Resting, B) Feeding, C) Pacing, and D) Object manipulation (interaction with enrichment items) throughout the habitat is characterized by the detection intensity colour gradient, where yellow and red regions indicate highly trafficked areas in association with the behaviour of interest, while blue and green regions represent lesser used areas.  


## Citation
```
Chen, L-D., Dodds, S., McGuire, M., Franke, M., & Mastromonaco, G. (2025). PantherAI: An autonomous behavioural monitoring tool for assessing activity budget and space use in a zoo-housed tiger. Ecological Informatics, 103584. https://doi.org/10.1016/j.ecoinf.2025.103584
```


## Contact
If you have any questions/ suggestions with the pipeline, or any additional instructions/ guidelines you would like to see in the documentation, feel free to contact me directly via email:

lchen[at]torontozoo.ca
