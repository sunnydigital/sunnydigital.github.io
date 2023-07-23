---
title: Discriminative Dimensioanlity Reduction via Learning a Tree
date: 2023-07-22T03:49:04-04:00
draft: true
ShowToc: true
math: true
tags: 
    - monocle
    - latent-model
    - dimensionality-reduction
cover:
    image: images/dalle-white-ball.png
    relative: true # To use relative path for cover image, used in hugo Page-bundles
header-includes:
    - \usepackage{lucidabr}
---

# Everything is Not What it (Quite) Seems

More than just a throwback to Wizards of Waverly Place, I really do think latent models are quite neat, the idea that there exists a fundamental discrepency between *what* is overtly shown and *how* these patterns came to be.

Nowhere is this as present as the complex systems involved in gene expression, where phenotypes (what characteristics we see on an organism, e.g. red hair and a lack of a soul expressed in gingers vs. black hair and a soul expressed in literally anyone else) are not always explained by genotypes (the literal ATCG in DNA). 

However in many cases across the medical field, the phenotypes of a specific organism (most of the time) might or might not make sense given the context of the genes at play.

As it turns out, the experimental observations of cellular expression (in terms of the genes to each of the cells) lie in a lower dimension (as a moment generating function) than the more complex assortment of observed expressions.

# When You Throw a Ball Into the Air, is it Really the Same Ball?

The analogy that helped me better understand how reverse graph embeddings operate was a simple one you can do at home, with your eyes closed. Imagine someone tossing a ball into the air, with you holding a camera pointing at the ball (tracing it's path through the air) from a third-person perspective off of the $z$-axis, taking snapshots every now and then, *snap*, *snap*, *snap*, *snap*, ... , *SNAP*. The film is developed, and out come five different photos.

You can imagine the snapshots as allegories of the ball stuck in time, with each picture giving context to where the ball is relative to everything around it. The photos look something like the following:

<figure id="fig1">
    <img src="images/fig-02.png" alt="Fig. 1">
    <figcaption align="center"><i>Fig. 1</i>. The five distinct photos taken of the ball on it's trajectory through the air... Ohhh <i>Snaap</i></figcaption>
</figure>

While you just took a total of five pictures, this number can be generalized to *N*, making it as arbitrarily large or small as you would like. From the instantaneous timeframe of these pictures, each one would be an accurate representation of where the ball was at that time. A more descriptive view can be shown through zooming out and looking at the larger picture:

<figure style="text-align: center;" id="fig2">
    <img src="images/fig-01.png" width="70%" alt="Fig. 2" style="display: inline-block;">
    <figcaption align="center"><i>Fig. 2</i>. Perspective from the z-axis of a ball being thrown into the air, with each ball representing a snapshot in time</figcaption>
</figure>

For clearer presentation, the previous two images can be concatenated - juxtaposing what can be considered the *discrete* timeframe of the photos you just took and the *continuous* live-demonstration of the ball being thrown. The below image demonstrates this, providing a graphical overlay of the photos at each time:

<figure style="text-align: center;" id="fig3">
    <img src="images/fig-03.png" width="70%" alt="Fig. 3" style="display: inline-block;">
    <figcaption align="center"><i>Fig. 3</i>. The connected image of both the photos and what is happening</figcaption>
</figure>

To put math towards the intuition of the previous images, we use a simple equation from kinematics representing a point-like object (which will be referred to as a ball from now on):

$$
x(t) = \frac{1}{2}a_it^2 + v_it + x_i, \ t \in \mathbb{R}; \ x_i, v_i, a_i \in \mathbb{R}^n
$$

where $x_i$ is the initial position of the ball, $v_i$ is the initial velocity of the ball, and $a_i$ is the acceleration of mass on earth (~$9.98 m/s$) (both variables in some *hypothetical* $n$-dimensional space), and $t$ is the time passed since the ball was first thrown up in the air.

For a brief moment, let's assume we know nothing about the *continuous* path that the ball takes, what we consider to be the closed-form solution of the position of the ball (equivalent to $x_i$). Instead let's now ask: how can we feasibly piece together the trajectory of the ball from the discrete images in the third figure?

To draw inspiration, we can look at the combined discrete/continuous image ([Fig. 3](#fig3)), which connects the time embedded and non-time embedded figures shown previously. For each of the smaller, discrete images, we can see that the most stark detail of difference (with the *ball* acting as a common reference frame throughout) is the background of each of the image.

Now what can be asked is: how do we put the question of a difference in background into a quantifiable detail? Well, the background is different only because of a change in the *position* of the ball (from *any* viewpoint of a 3rd party observer). This intuition is *EXACTLY* how graph dimensionality-reduction learning operates at a high level: to find the *path* of the ball (the generalized equation of sorts) that best describes all of the images of the ball, given its position (the discrete images of the ball) at different timepoints.

This is the loose analogy between a simpler, physical representation of the mechanisms at play and what the process is actually explaining (a biological one). Through generalization, we can finally get closer to what the algorithm was meant to represent: cells and expressed genes. Across all of our samples of cells with genes as features, expression can be interpreted as pictures of individual cells and the latent distribution of *possible* outcomes (moderated by time) as the continuous point of view.

# Learning High & Low - Surfing Both Oceans and Webs

While the previous analogy works well as a high-level concept, it is far too *deterministic*, as given the parameters $x_i$, $v_i$, and $a_i$, the height $x(t)$ can be calculated precisely. This is not the case with the randomness present in our genotype vs. phenotype dilemma.

To explain, we need to draw our analogy out futher through taking inspiration from nature, particularly waves and water:

<figure style="text-align: center;" id="fig4">
    <img src="images/fig-04.gif" width="70%" alt="Fig. 4" style="display: inline-block;">
    <figcaption align="center"><i>Fig. 4</i>. Water waves with refractions as seen from above</figcaption>
</figure>

While the image is pretty to say the least, there does exist a degree of allegorical importance separate from the aesthetic. The sort of randomness present in the waves expressed through the height of the *pish, pash, splatter* of the water, alongside the seemingly continuous nature of the *fabric* at the surface of the liquid, is closer to the nature of the genes expressed (phenotype) than a simple ball toss.

But then what of the genotype? In some terms, the genotype can bs seen as a spider web, still yet fluctuating against the force of wind, but never so sporadic as to leave parts in tandem with unseen possibility. In some terms, it is far more stable than the phenotype, given the codified nature of DNA, but also *simpler* given a substantial lack of variation between one instance of genotype and another (as in a comparison between the expression in cells).

<figure style="text-align: center;" id="fig5">
    <img src="images/fig-05.gif" width="50%" alt="Fig. 5" style="display: inline-block;">
    <figcaption align="center"><i>Fig. 5</i>. The simplicity of a spider web in comparison to water waves</figcaption>
</figure>

Now, with a more comprehensive grasp of this allegory, we can begin to add mathematical concepts and numbers in our description.

# The Mountain Range (High and Low)

As described previously, waves can be used to represent the high-dimensional output of genes that are actually expressed as a phenotype.

<figure style="text-align: center;" id="fig6">
    <img src="images/fig-06.png" width="50%" alt="Fig. 6" style="display: inline-block;">
    <figcaption align="center"><i>Fig. 6</i>. Our mathematical model of high-dimensional waves</figcaption>
</figure>

In this mathematically analogous model, we assume there is some way to ordinally number all states of the expressed genes (phenotypes) in the high dimensional space, set as the variable $\mathcal{X}=\{\mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_N\}$, where $N \in \mathbb{I}$ is the total number of cells. This is essentially listing out the dimensions involved with each cell (i.e. the vector of genes expressed per cell) for every cell in the sampled tissue.

<figure style="text-align: center;" id="fig7">
    <img src="images/fig-07.png" width="50%" alt="Fig. 7" style="display: inline-block;">
    <figcaption align="center"><i>Fig. 7</i>. Our mathematical model of low-dimensional webs</figcaption>
</figure>

In the lower-dimensional space, we define the variables associated with the diversity of gene expressions (genotypes) across individual cells as $\mathcal{Z}=\{\mathbf{z}_1, \mathbf{z}_2,...,\mathbf{z}_N\}$, where $N \in \mathbb{I}$ is the total number of cells. However, in this context, the representation of each variable is less discernible. They are composed of *latent* (unseen) nodes, each depicting a cell's true phenotype generating distribution. In other words, these latent variables are not directly observed but are inferred from the observed gene expressions. They serve as a representation of the underlying biological states/properties of each cell.

## A Map(ping) to Guide Us

We are now just about ready to define the structured connection between web and the wave, a low-dimensional insight into a high-dimensional output. To start, we must image *how* we define the low-dimensional web we want to describe, in other words the abstracted qualities the web should have. Looking at [Fig. 7](#fig7), some initial qualities of note can be pointed out:

1. Circular parts that contain each of the cells, which we will refer to as *nodes* from now on
2. Lines connecting those circles, acting as *edges*
3. In something combined that is referred to as a *graph*

In more mathematical notation, we can now say that the Graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$ contains a set of vertices $\mathcal{V}=\{\mathcal{V}_1, \mathcal{V}_2,...,\mathcal{V}_N\}$ and a wet of weighted, undirected edges $\mathcal{E}$. This stands in stark contast to unweighted and directed edges, where the former lacks a representation for the numerical degree of connection between any two vertices, and the latter meaning for a sense of directionality or pointing from one node vertex to another.

# Reverse Graph Embeddings

As the actual map(ping) from the higher dimension to the low, we use somethign called Reverse Graph Embeddings. The equation is as follows:

$$
\argmin_{\mathcal{G} \in \mathcal{G}_b} \argmin_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}} \argmin_{Z} \sum_{(V_i, V_j) \in \mathcal{E}} b_{i,j} || \mathcal{f}_{\mathcal{G}}(\mathbf{z}_i)-\mathcal{f}_{\mathcal{G}}(\mathbf{z}_j)||^2
$$

## Arrrgh (mins)

This one is a doozy, and seemingly so frustrating, so let's break it down piece by digestible piece. We are initially met with three $\argmin$'s: 

$$
\argmin_{\mathcal{G} \in \mathcal{G}_b} \argmin_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}} \argmin_{Z}
$$

In the form of a triple nested optimization problem, which is common in machine learning when we're trying to find the best model parameters that minimize a certain loss function. Let's start from the inside and work our way out:

1. The First $\argmin$ selecting the Optimal Graph $\mathcal{G}$: 

    $$
    \argmin_{\mathcal{G} \in \mathcal{G}_b}
    $$

    At the core of Reverse Graph Embeddings lies a graph $\mathcal{G}$ that captures the relationships between data points. However, not all graphs are created equal. The first $\argmin$ allows us to explore different graphs within the set $\mathcal{G}_b$ and identify the one that best represents the underlying structure of the data.

2. The Second $\argmin$ optimizing the Embedding Function $\mathcal{f}{\mathcal{G}}$

    $$
    \argmin_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}}
    $$

    Having chosen the optimal graph $\mathcal{G}$, we now need to determine the most suitable embedding function $\mathcal{f}{\mathcal{G}}$. This function maps the data points from the higher-dimensional space to a lower-dimensional space, where relationships between points are preserved and the graph structure is maintained. Through the second $\argmin$, we search for the best $\mathcal{f}_{\mathcal{G}}$ from the set $\mathcal{F}$ of possible functions.

1. The Third $\argmin$ finds the Optimal Embeddings $Z$:
   
    $$
    \argmin_{Z}
    $$

   With the graph $\mathcal{G}$ and the embedding function $\mathcal{f}_{\mathcal{G}}$ in place, the third $\argmin$ seeks to identify the optimal low-dimensional embeddings $Z$. These embeddings are representations of the data points in the lower-dimensional space that minimize the loss function defined by the summation in the equation.

## Simplified Loss Function

Now, we get to the simplified loss function, or the part of the equation stated as:

$$
\sum_{(V_i, V_j) \in \mathcal{E}} b_{i,j} || \mathcal{f}_{\mathcal{G}}(\mathbf{z}_i)-\mathcal{f}_{\mathcal{G}}(\mathbf{z}_j)||^2
$$

The objective of the loss function is to measure the similarity or dissimilarity between two data points in the graph $\mathcal{G}$. As we traverse the graph's edges represented by $(V_i, V_j)$ in the set $\mathcal{E}$, the loss function computes the difference between the embeddings of these connected data points.

Graph-Based Weights:

The loss function incorporates the notion of graph-based weights $b_{i,j}$. These weights allow us to assign different levels of importance to the edges connecting data points within the graph $\mathcal{G}$. By introducing these weights, we can emphasize or de-emphasize certain relationships, depending on their significance in the overall data representation.

Optimizing Low-Dimensional Embeddings:

The central aim of Reverse Graph Embeddings is to minimize the loss function. As the triple nested optimization progresses, the graph $\mathcal{G}$ and the embedding function $\mathcal{f}_{\mathcal{G}}$ are fine-tuned to achieve the optimal low-dimensional embeddings $Z$. These embeddings represent the data in a compact and meaningful manner in the lower-dimensional space.

Preserving Graph Structure:

Through the process of minimizing the loss function, the algorithm ensures that the relationships between data psaoints observed in the original graph $\mathcal{G}$ are maintained in the lower-dimensional space. This preservation of graph structure is essential for extracting meaningful insights and knowledge from the data.

## Complete Loss Function and Visualization

This is a simplified version of the complete Reverse Graph Embedding. This previous example *only* considers graph structures in the latent (genotype) state, but not the observed phenotypes within the optimization parameters. To find a way to connect the high and low dimensions, RGE both ensures that

1. The image under the function $\mathcal{f}_{\mathcal{G}}$ (points in the high-dimensional space as a *function* of the low-dimensional space) are close to one another as we described previously, save for the $\lambda$ hyperparameter which is soon to be explained:

$$
\frac{\lambda}{2}\sum_{(V_i, V_j) \in \mathcal{E}} b_{i,j} || \mathcal{f}_{\mathcal{G}}(\mathbf{z}_i)-\mathcal{f}_{\mathcal{G}}(\mathbf{z}_j)||^2
$$

2. An addended portion states that points which are neighbors on the low-dimensional principal graph are also "neighbors" in the input dimension, meaning for a given $\mathbf{z}_i$ (where $i$ is the data associated with a specific cell), the *estimated* phenotypes expressed must also be similar to the *real* phenotypes expressed by the cell in the high-dimension:

$$
\sum_{i=1}^{N} || \mathbf{x}_i - \mathcal{f}_{\mathcal{G}}(\mathbf{z}_i) ||^2
$$

Resulting in the combined equation:

$$
\argmin_{\mathcal{G} \in \mathcal{G}_b} \argmin_{\mathcal{f}_{\mathcal{G}} \in \mathcal{F}} \argmin_{Z} \sum_{i=1}^{N} || \mathbf{x}_i - \mathcal{f}_{\mathcal{G}}(\mathbf{z}_i) ||^2 + \frac{\lambda}{2}\sum_{(V_i, V_j) \in \mathcal{E}} b_{i,j} || \mathcal{f}_{\mathcal{G}}(\mathbf{z}_i)-\mathcal{f}_{\mathcal{G}}(\mathbf{z}_j)||^2
$$

Altogether, the below figure best summarizes what is taking place:

<figure style="text-align: center;" id="fig8">
    <img src="images/fig-08.png" width="100%" alt="Fig. 8" style="display: inline-block;">
    <figcaption align="center"><i>Fig. 8</i>. Our completed model of waves and webs</figcaption>
</figure>

Where some latent graph of points (some genotype) is found to map onto the observed genotypes of the cells in such a way as to follow the constraints set by the combined equation.

# A Conclusion, Or.. Is It?

With this, we have finally finished our coverage of the theory behind using graph-based online methods for dimensionality reduction. However, the main paper on monocle (which the authors have graciously allowed to be released [here](https://cole-trapnell-lab.github.io/pdfs/papers/qiu-monocle2.pdf)) does delve deeper into how *trees* are utilized to draw a pseudo-temporal mapping of cells in different states of division, that will be saved for another post!