---
title: IEGSIL
date: 2023-07-02T03:49:04-04:00
draft: false
ShowToc: true
cover:
  image: iegsil-demo-pic.jpg
  # can also paste direct link from external site
  # ex. https://i.ibb.co/K0HVPBd/paper-mod-profilemode.png
  alt: Photo of completed build
  caption:
  relative: false # To use relative path for cover image, used in hugo Page-bundles
---

## Why not?

In all honesty, whilst building RIEGSIL v2 for ML applications, I just happened to have left over graphics processors, a CPU, and a power supply

With the GPUs having already been fitted with waterblocks it was near impossible to sell, so in the interest of reusability, I headed over to the local Micro Center and bought a few parts to assemble a smaller form-factor PC than the original RIEGSIL v1.

This was quite the process as it turned out:

![](=iegsil-demo-pic.jpg)

## Parts list

This is an exhaustive list of parts used in the build

| Type | Item | Link |
| :---- | :----: | ----: |
| **CPU** | AMD Ryzen 9 3900X 3.8 GHz 12-Core Processor | [@ Amazon](https://www.amazon.com/dp/B07SXMZLP9?tag=pcpapi-20&linkCode=ogi&th=1&psc=1) |
| **Motherboard** | ASRock X570M Pro4 Micro ATX AM4 Motherboard | [@ Newegg](https://www.newegg.com/asrock-x570m-pro4/p/N82E16813157887) |
| **Memory** | 2x G.Skill Trident Z Royal 32 GB (2 x 16 GB) DDR4-3600 CL19 Memory | [@ Amazon](https://www.amazon.com/dp/B07SQ3T4X3?tag=pcpapi-20&linkCode=ogi&th=1&psc=1) |
| **Storage** | 2x Seagate IronWolf NAS 12 TB 3.5" 7200 RPM Internal Hard Drive | [@ Newegg](https://www.newegg.com/seagate-ironwolf-st12000vn0008-12tb/p/1JW-001N-00027?Item=1JW-001N-00027&nm_mc=AFC-RAN-COM&cm_mmc=afc-ran-com-_-PCPartPicker&utm_medium=affiliate&utm_campaign=afc-ran-com-_-PCPartPicker&utm_source=afc-PCPartPicker&AFFID=2558510&AFFNAME=PCPartPicker&ACRID=1&ASID=https%3a%2f%2fpcpartpicker.com%2f&ranMID=44583&ranEAID=2558510&ranSiteID=8BacdVP0GFs-HPYWcPc.wqLNeIWmTYX2aQ) |
| **Video Card** | 2x NVIDIA TITAN RTX TITAN RTX 24 GB Video Card | [@ Amazon](https://www.amazon.com/dp/B07L8YGDL5?tag=pcpapi-20&linkCode=ogi&th=1&psc=1) |
| **Cables** | ARGB Hub 5V 3Pin SYNC 11 Ports Splitter w/ Magnetic Standoff & PMMA Case,SATA to 3pin Addressable RGB Adpater | [@ Newegg](https://www.newegg.com/p/1W7-005X-00093?Item=9SIACJFCAZ4145) |
| **Networking** | Intel WiFi 6 AX200 Gig M2.2230 Kit | [@ Micro Center](https://www.microcenter.com/product/636193/intel-wifi-6-ax200-gig-m22230-kit) |
| **Waterblock** | 2x Bitspower GPU Waterblock for NVIDIA GeForce RTX 2080Ti / 2080 Reference Cards with Accessory Set | [@ Newegg](https://www.newegg.com/p/37B-000X-003Y5?Item=9SIAEMWAZE0901) |
| **Watercooling** | Bitspower G1/4" Adjustable Aqua Link Pipe (41-69mm), Silver Shining | [@ Newegg](https://www.newegg.com/bitspower-bp-dg14aalpii-fittings/p/2YM-0001-00012?Item=9SIAEKS6MM8126) |
| **Case & Watercooling** | Bitspower TITAN ONE MINI 2.0 | [@ MicroCenter](https://www.microcenter.com/product/661034/bitspower-titan-one-mini-20-tempered-glass-microatx-mini-tower-computer-case-black) |
| **Power Supply** | Silverstone SX1000-LPT 1000 W 80+ Platinum Certified Fully Modular SFX Power Supply | [@ Micro Center](https://www.google.com/search?q=silverstone+1000w+sfx+psu&rlz=1C1QMKX_enUS1049US1049&oq=silverstone+sff+psu+1000&gs_lcrp=EgZjaHJvbWUqCggBEAAYCBgNGB4yBggAEEUYOTIKCAEQABgIGA0YHtIBCDczNTdqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8) |

## Getting started

The first step involved a bit of cleaning, actually - a lot of cleaning. 

Due to the Titan RTX GPUs having been removed from their original [EKWB](https://www.ekwb.com/) waterblocks used in RIEGSIL v1 (post coming shortly), and repurposed with [Bitspower](https://bitspower.com/) waterblocks I bought on sale (thanks to the outdated nature of the Titan RTX), extensive cleaning was needed on the GPU chip.

Wanting to try an new PC parts company and utilizing that company's parts throughout the build, I settled on Lian Li's [TITAN ONE MINI 2.0](https://bitspower.com/titanseries/titan_one_mini_2.0/) for its simplicity and integration with Bitspower watercooling parts, as well as innovative daisychain [Uni Fan SL v2s](https://lian-li.com/product/uni-fan-sl-v2/), of which there are 7x 120mm and 2x 140mm (for side ventilation).

