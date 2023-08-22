---
title: The Quantization Conundrum or, Softmax Slapstick
date: 2023-08-16T07:00:00-04:00
draft: true
ShowToc: true
math: true
tags: 
    - softmax
    - LLM
    - quantization
cover:
    image: images/quantization.jpg
    relative: true # To use relative path for cover image, used in hugo Page-bundles
---

## Our Limited World of Limited Scope

In our digital age, everything we see being displayed on our screens consists of 0's and 1's, or *bits*. However, when contrasted to the real world, these digital representations often have limitations in scope and capability of implementation.

To illustrate this, let's consider the following example. Suppose we have a 1D array of 8 bits, or *bytes*, and we want to represent the number 255. In binary, this number is represented as 11111111. However, if we were to represent this number in our 1D array, we would have to use all 8 bits, and we would have no room for any other numbers. This is because our array is limited in scope, and can only hold 8 bits. If we wanted to represent the number 256, we would need 9 bits, and our array would not be able to hold it. This is a limitation of our digital world, and it is something that we must keep in mind when designing algorithms and systems.

## Quantization Crash Course

To explore quantization, lets begin with the different kinds of numerical representations available to 