---
title: Discriminative Dimensioanlity Reduction via learning a Tree
date: 2023-07-10T03:49:04-04:00
draft: true
ShowToc: true
cover:
  image: images/dalle-white-ball.png
  relative: true # To use relative path for cover image, used in hugo Page-bundles
---

# Everything is not what they seem

I think latent models are quite neat, the idea that there exists a *fundamental* discrepency between what is overtly shown and *how* such patterns came to be.

Nowhere is this as present as the complex systems involved in gene expression, where the phenotypes we see across an organism might (most of the time) or might not make sense given the context of the genes at play.

As it turns out, experimentally the observations of cellular expression (in terms of the genes to each of the cells) lie in a lower dimension (as a generating distribution) than the more complex assortment of observed expression outcomes.

---
***NOTE***
The previous part needs revision, in terms of the cellular expression (genes to each of the cells)
---

# When you throw a ball into the air, is there a different ball every time you blink?

The analogy that helped me better understand how reverse graph embeddings operate was a simple one you can do at home, with your eyes closed. When you toss a ball into the air, imagine a camera pointing at the ball from your point of view, taking snapshots every now and then, *snap*, *snap*, *snap*, ... , *SNAP*. 

[INSERT IMAGE OF MULTIPLE SHOTS OF A BALL THROWN AND FALLING]

I just took a total of *N* pictures (make this number as arbitrarily large or small as you would like), and from the instantaneous timeframe of the picture, each would be an accurate representation of where the ball was at that time and as such, the representation would then be:

[INSERT GRAPH OF SNAPSHOTS OF BALL AT DIFFERENT TIMESTAMPS, WITH A STICK FIGURE VIEWING THE PROGRESSION]

To put math towards the intuition of images, we use a simple equation 