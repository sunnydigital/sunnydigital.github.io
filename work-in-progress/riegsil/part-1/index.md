---
title: RIEGSIL | Part 1
date: 2023-07-06T07:00:00-04:00
draft: false
ShowToc: false
cover:
  image: images/riegsil-wip.jpg
  # alt: A next-gen DL workstation
  caption: RIEGSIL as a work-in-progress
  # relative: false # To use relative path for cover image, used in hugo Page-bundles
---

## Part 1

This post summarizes the first part of the RIEGSIL build.

It starts off from a barebones chassis and enumerates through to fan installation, modifying and installing the motherboard, and finishes on the connecting most I/O and necessary ports.

## Getting started

First, the motherboard was prepared by carefully installing the CPU (the renowned Threadripper 3990X from AMD, allowing up to 64-core 128-thread performance), and 5x M.2 2TB hard drives in RAID 0 (backed up, of course, on hard drives). The CPU was chosen for the future optionality to dedicate a part of this build as a server or remote compute device.

<figure>
    <img src="images/motherboard-thermal-pads.jpg" alt="Zenith II Extreme Alpha without attachments">
    <figcaption align="center">The motherboard, installed CPU, and two M.2 hard drives</figcaption>
</figure>

Then, the monoblock (a waterblock for a CPU that also covers the chipsets of the motherboard) was also installed, but not before much careful measuring and cutting of thermal pads. This monoblock replaces the motherboard fan delivering cooling to the chipset, allowing for extended heat-dissapation.

<figure>
    <img src="images/motherboard-monoblock.jpg" alt="Zenith II Extreme Alpha with thermal pads">
    <figcaption align="center">The motherboard, installed CPU, M.2 drives, and monoblock</figcaption>
</figure>

Finally, 8x 32GB RAM sticks were installed, to provide fast DDR4 speeds for calculations, and memory for multiple instances of programs and computer vision requirements.

<figure>
    <img src="images/motherboard-full.jpg" alt="Zenith II Extreme Alpha with thermal pads">
    <figcaption align="center">The fully assembled motherboard with monoblock and RAM</figcaption>
</figure>

For the GPUs, two 4090's were chosen for their performance. Although SLI/NV-LINK is not present for this generation of NVIDIA graphics processors, multi-GPU processing in Pytorch (the primary) 

<figure>
    <img src="images/4090s-no-waterblock.jpg" alt="4090s with air coolers">
    <figcaption align="center">Two Strix 4090s, with their <i>massive</i> air coolers</figcaption>
</figure>

As with the motherboard, the massive stock cooler (heatsink) was removed and waterblocks by EKWB installed in their place:

<figure>
    <img src="images/4090-circuits.jpg" alt="4090 without air cooler">
    <img src="images/4090-thermal-pads.jpg" alt="4090 with thermal pads attached">
    <figcaption align="center">Before and after of a 4090 without the stock air cooler</figcaption>
</figure>

Et, voilà: a pair of cool, black, waterblock-fitted 4090s!

<figure>
    <img src="images/4090s-waterblock.jpg" alt="4090s with waterblocks">
    <figcaption align="center">Two Strix 4090s, form fitted with waterblocks</figcaption>
</figure>

To be continued, as I take the parts from the desk top (PC pun) to the case!