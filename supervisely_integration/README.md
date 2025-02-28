
<div align="center" markdown>

<img src="https://github.com/user-attachments/assets/1cb32840-6096-48aa-8b2b-1663d822a72c" style="width: 70%;"/>

# MCITrack Object Tracking

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Controls">Controls</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/mcitrack-object-tracking)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mcitrack)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/mcitrack/supervisely_integration.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/mcitrack/supervisely_integration.png)](https://supervisely.com)

</div>

# Overview

MCITrack architecture consists of three parts: a backbone for visual feature extraction, a Contextual Information Fusion (CIF) module for storing and transmitting contextual information, and a prediction head for making predictions. MCITrack takes a video clip and a search region as input. First, the video clip and search region are divided into patches through patch embedding, and then these patches are concatenated along the spatial dimension and fed into the backbone for feature extraction. The backbone is composed of N blocks, with each block paired with a corresponding CIF block. The CIF blocks integrate the historical contextual information into their associated backbone blocks, enhancing the accuracy of visual feature extraction based on the historical contextual information. Simultaneously, the CIF blocks update the hidden states based on the current backbone output. Finally, the backbone, guided by contextual information, outputs more precise visual features, which are then passed to the prediction head to obtain the tracking results. The head consists of three sub-networks, each composed of convolutional layers.

![architecture](https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/mcitrack_architecture.png)

MCITrack demonstrates strong performance on various visual object tracking benchmarks, achieving state-of-the-art performance on LaSOT and TrackingNet datatsets:

![performance](https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/mcitrack_performance.png)

# How To Run

**Step 1.** Select pretrained model architecture and press the **Serve** button

![pretrained_models](https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/mcitrack_deploy.png)

**Step 2.** Wait for the model to deploy

![deployed](https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/mcitrack_deploy_2.png)

# Controls

| Key                                                           | Description                               |
| ------------------------------------------------------------- | ------------------------------------------|
| <kbd>5</kbd>                                       | Rectangle Tool                |
| <kbd>Ctrl + Space</kbd>                                       | Complete Annotating Object                |
| <kbd>Space</kbd>                                              | Complete Annotating Figure                |
| <kbd>Shift + T</kbd>                                          | Track Selected     |
| <kbd>Shift + Enter</kbd>                                      | Play Segment     |

# How To Use

Create input bounding box, select desired number of frames to track and press "Track all on screen":

<video width="100%" preload="auto" autoplay muted loop>
    <source src="https://github.com/supervisely-ecosystem/MCITrack/releases/download/v0.0.1/mcitrack_example.mp4" type="video/mp4">
</video>

# Acknowledgment

This app is based on the great work [MCITrack](https://github.com/kangben258/MCITrack). ![GitHub Org's stars](https://img.shields.io/github/stars/kangben258/MCITrack?style=social)
