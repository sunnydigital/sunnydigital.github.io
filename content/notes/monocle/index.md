---
title: Discriminative Dimensioanlity Reduction via Learning a Tree
date: 2023-07-10T03:49:04-04:00
draft: true
ShowToc: true
math: true
cover:
  image: images/dalle-white-ball.png
  relative: true # To use relative path for cover image, used in hugo Page-bundles
---

# Everything is not what it seems

I think latent models are quite neat, the idea that there exists a fundamental discrepency between *what* is overtly shown and *how* such patterns came to be.

Nowhere is this as present as the complex systems involved in gene expression, where the phenotypes we see across an organism might (most of the time) or might not make sense given the context of the genes at play.

As it turns out, the experimental observations of cellular expression (in terms of the genes to each of the cells) lie in a lower dimension (as a moment generating function) than the more complex assortment of observed expressions.

# When you throw a ball into the air, is it actually a different ball every time you blink?

The analogy that helped me better understand how reverse graph embeddings operate was a simple one you can do at home, with your eyes closed. Imagine someone tossing a ball into the air, with you holding a camera pointing at the ball (tracing it's path through the air) from a third-person perspective off of the $z$-axis, taking snapshots every now and then, *snap*, *snap*, *snap*, *snap*, ... , *SNAP*. The film is developed, and out come five different photos.

You can imagine the snapshots as allegories of the ball stuck in time, with each picture giving context to where the ball is relative to everything around it. The photos look something like the following:

<figure>
    <img src="images/fig-02.png" alt="Figure 1">
    <figcaption align="center"><i>Figure 1</i> The five distinct photos taken of the ball on it's trajectory through the air... OHhh <i>SNAap</i></figcaption>
</figure>

While you just took a total of five pictures, this number can be generalized to *N*, making it as arbitrarily large or small as you would like. From the instantaneous timeframe of these pictures, each one would be an accurate representation of where the ball was at that time. A more descriptive view can be shown through zooming out and looking at the larger picture:

<figure>
    <img src="images/fig-01.png" alt="Figure 2">
    <figcaption align="center"><i>Figure 2</i> Perspective from the z-axis of a ball being thrown into the air, with each ball representing a snapshot in time</figcaption>
</figure>

These two images can then be connected, from discrete snapshots of the ball in motion to the continuous flow of the ball as shown below:

<figure>
    <img src="images/fig-03.png" alt="Figure 3">
    <figcaption align="center"><i>Figure 3</i> The connected image of both the photos and what is happening</figcaption>
</figure>

To put math towards the intuition of images, we use a simple equation from kinematics representing a point-like object (which will be referred to as a ball from now on):

$$
x(t) = vt + x_i, \ t \in \mathbb{R}; \ p,v \in \mathbb{R}^n
$$

where $x_i$ is the initial position of the ball, $v$ is the velocity of the ball (both in some *hypothetical* $n$-dimensional space), and $t$ is the time passed since the ball was first thrown up in the air.

For a brief moment, let's assume we know nothing about the *continuous* path that the ball takes, what we consider to be the closed-form solution of the position of the ball (equivalent to $x_i$). Instead let's now ask: how can we feasibly piece together the trajectory of the ball from the discrete images in the third figure?

To draw inspiration, we can look at the combined discrete/continuous image (figure 3), which connects the time embedded and non-time embedded figures shown previously. For each of the smaller, discrete images, we can see that the most stark detail of difference (with the *ball* acting as a common reference frame throughout) is the background of each of the image.

Now what can be asked is: how do we put the question of a difference in background into a quantifiable detail? Well, the background is different only because of a change in the *position* of the ball (from *any* viewpoint of a 3rd party observer). This intuition is *EXACTLY* how graph dimensionality-reduction learning operates at a high level: to find the *path* of the ball (the generalized equation of sorts) that best describes all of the images of the ball, given its position (the discrete images of the ball) at different timepoints.

This is the loose analogy between a simpler, physical representation of the mechanisms at play and what the process is actually explaining (a biological one). Through generalization, we can finally consider what the algorithm was meant to represent: cells and expressed genes. Across all of our given samples of cells with genes as features, the expressed genes can be interpreted as pictures of individual cells, 