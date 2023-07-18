---
title: Discriminative Dimensioanlity Reduction via learning a Tree
date: 2023-07-10T03:49:04-04:00
draft: false
ShowToc: true
math: true
cover:
  image: images/dalle-white-ball.png
  relative: true # To use relative path for cover image, used in hugo Page-bundles
---

# Everything is not what they seem {style="text-aligh: justify;"}

I think latent models are quite neat, the idea that there exists a *fundamental* discrepency between what is overtly shown and *how* such patterns came to be.

Nowhere is this as present as the complex systems involved in gene expression, where the phenotypes we see across an organism might (most of the time) or might not make sense given the context of the genes at play.

As it turns out, experimentally the observations of cellular expression (in terms of the genes to each of the cells) lie in a lower dimension (as a generating distribution) than the more complex assortment of observed expression outcomes.

---
***NOTE***
The previous part needs revision, in terms of the cellular expression (genes to each of the cells)
---

# When you throw a ball into the air, is there a different ball every time you blink?

The analogy that helped me better understand how reverse graph embeddings operate was a simple one you can do at home, with your eyes closed. Imagine tossing a ball into the air, with a camera pointing at the ball from a third-person perspective off of the $z$-axis, taking snapshots every now and then, *snap*, *snap*, *snap*, *snap*, ... , *SNAP*. 

[INSERT IMAGE OF MULTIPLE SHOTS OF A BALL THROWN AND FALLING]

I just took a total of *N* pictures (make this number as arbitrarily large or small as you would like), and from the instantaneous timeframe of the picture, each would be an accurate representation of where the ball was at that time and as such, the representation would then be:

[INSERT GRAPH OF SNAPSHOTS OF BALL AT DIFFERENT TIMESTAMPS, WITH A STICK FIGURE VIEWING THE PROGRESSION]

These two images can then be connected, from discrete snapshots of the ball in motion to the continuous flow of the ball as shown below:

[OVERLAID IMAGES OF THE BALL IN MOTION AND INDIVIDUAL SNAPSHOTS OF THE BALL]

To put math towards the intuition of images, we use a simple equation from kinematics representing a point-like object (which will be referred to as a ball from now on):

$$
x(t) = vt + x_i, t \in \mathbb{R} p,v \in \mathbb{R}^n
$$

where $x_i$ is the initial position of the ball, $v$ is the velocity of the ball (both in some *hypothetical* $n$-dimensional space), and $t$ is the time passed since the ball was first thrown up in the air.

For a brief moment, let's assume we know nothing about the *continuous* path that the ball takes, what we consider to be the closed-form solution of the position of the ball (equivalent to $x_i$). Instead let's now ask: how can we feasibly piece together the trajectory of the ball from the discrete images in the third figure?

To draw inspiration, we can look at the combined discrete/continuous image (figure 3), which connects the time embedded and non-time embedded figures shown previously. For each of the smaller, discrete images, we can see that the most stark detail of difference (with the *ball* acting as a common reference frame throughout) is the background of each of the image.

Now what can be asked is: how do we put the question of a difference in background into a quantifiable detail? Well, the background is different only because of a change in the *position* of the ball (from *any* viewpoint of a 3rd party observer). This intuition is *EXACTLY* how graph dimensionality-reduction learning operates at a high level: to find the *path* of the ball (the generalized equation of sorts) that best describes all of the images of the ball, given its position (the discrete images of the ball) at different timepoints.

To generalize, we can consider what the algorithm was meant to describe: cells and expressed genes.