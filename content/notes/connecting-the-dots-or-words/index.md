---
title: Connecting the Dots (or Words)
date: 2023-08-29T07:00:00-04:00
draft: true
ShowToc: true
math: true
tags: 
    - LLM
    - language
    - n-dimensional
    - neural-networks
    - connecting
    - dots
cover:
    image: images/connecting-the-dots.jpg
    relative: true # To use relative path for cover image, used in hugo Page-bundles
---

## Are words, so..?

I had a very random thought the other day when writing a short piece. Perhaps it really is a "me" problem, but I was never one to flush out the words in an essay without demurring for a period of time to piece together partials of ideas to be stitched together later on. Although this may be contrary to what is presented when orating, planning has generally been every poet's friend.

This is currently not the case with LLMs in the context of machine learning. The general concensus utilizes a process that is trained to predict the next word in a sequence of words given a past series of words. In reality, the described model is more sophisticated in two ways: one, a so-called sub-word tokenization, where words are broken down into smaller units (a great application to try this out yourself can be found [here](httos://sunnyson.dev)) and two, the tokenizers are then actually mapped to cardinal numbers (1, 2, 3, 4, etc.)

But back to the point. Why should modern models only try to predict the next *token* given a series of previous ones in this autoregressive manner, given that when writing, we prepare and research - an iterative process with a non-fixed route?

## Setting the stage

To delve into our discussion of language models, we must first briefly consider how machines truly *understand* words. It may seem dualy ironic for a glorified piece of rock to speak, but bear with me.

We would first need a way to represent words as numbers. However, the problem arises when we note that words themseleves do not hold any intrinsic *value* (in the sense that numbers provide), but rather act as independent *indices* to denote a discrete set of concepts. This is where the concept of *embedding* comes in. In the context of machine learning, an embedding is the process of mapping a discrete set of concepts to a continuous space.

In the case of words, this is done by training a model to predict the next word in a sequence given a series of previous words. The model is then trained to minimize the error between the predicted word and the actual word. The model is then able to learn the relationship between words and their context, and thus, the embedding is created.

We implement this through the use of a pretrained *embedding*, which for the purposes of this article, simply creates a vocabulary (set of words), quantizing each word (mapping each to an integer) and finally projects all the words into a lower dimension (GloVe-50d does just this in 50 dimensions). Finally, to visualize this in the 30dimensions available to us, we use dimensionality reduction to project the total number of dimensions to 3.

{{< gist spf13 7896402 >}}

The below diagram shows a hypothetical embedding of words in a 3-dimensional space made using the above code.

In effect, the embedding takes a vocabulary (set) of words and through tokenization, indexes each to an integer. 

## An arbitrary path of thought

Let us consider an analogous example. Below is a figure representing words available in some arbitrary dimension *n*. This dimensionality is arbitrary for now, given the unknown nature of what words can represent and entail in our mental model.

<figure style="text-align: center;" id="fig-1">
    <img src="images/fig-1.gif" width="70%" alt="Fig. 1" style="display: inline-block;">
    <figcaption align="center"><i>Fig. 1</i>. Our reduced word model, implementing a 3d representation of an arbitrary number of dimensions for each word</figcaption>
</figure>

The dots in this case represent individual words in a high dimensional space that has been projected into 3-d for comprehension purposes.

Now, let's apply the creation of the phrase:

$$ \texttt{Let's go to the beach, I heard the weather's nice} $$

For the sake of simplicity, we can assume tokenization on the single-word level, providing us with the below format of sequence, where we implement the special tokens $\texttt{<BOS>}$ and $\texttt{<EOS>}$ to represent the beginning of a sequence and the end of a sequence respectively. This transforms the tokenizer, vocabulary, and associated phrase into the following:

$$ \texttt{[<BOS>]} \rightarrow \texttt{[Let's]} \rightarrow \texttt{[go]} \rightarrow \texttt{[to]} \rightarrow \texttt{[the]} \rightarrow \texttt{[beach]} \rightarrow \texttt{[I]} \rightarrow \texttt{[heard]} \rightarrow \texttt{[the]} \rightarrow \texttt{[weather's]} \rightarrow \texttt{[nice]} \rightarrow \texttt{[<EOS>]} $$

To apply our model, the first dimension will be made into each a representation of the order present for the sequence we are considering. Furthermore, we will have the simplyfing assumption that for dimension 2, we will have the starting word for that order in the sequence we are considering, and for dimension 3, we will have the ending word for that order in the sequence we are considering. While this fills up the space provided by each dimension, please remember that still, this is a simple representation of an arbitrarily high-dimensional space representing the *meaning* for each word.

At the same time, let's consider the amended phrase: 

$$ \texttt{Let's... go to... beach... weather's... nice} $$ 

While the words are still technically in the same order, intermediarily, connective nodes are missing, and the words involved in producing a coherent meaning as in the first example are not decisively defined. This would be the same as our previous example if we had not planned our writing beforehand, but rather relied on the inclusion of the above words or concepts (the latter deals with a potentially more excessive discussion than I can provide for at the moment). The below is effectively the token transition sequence:

$$ \texttt{[<BOS>]} \rightarrow \texttt{[Let's]} \rightarrow \texttt{[<CON>]} \rightarrow \texttt{[go]} \rightarrow \texttt{[to]} \rightarrow \texttt{[<CON>]} \rightarrow \texttt{[beach]} \rightarrow \texttt{[<CON>]} \rightarrow \texttt{[weather's]} \rightarrow \texttt{[<CON>]} \rightarrow \texttt{[nice]} \rightarrow \texttt{[<EOS>]} $$

Where the $\texttt{<CON>}$ token is a placeholder for any length of missing words. It is therefore special as the only token in our current model to be able to act as a placeholder for multiple words.

In our simplified word model the complete transition sequences and partial transition sequences can be represented graphically as:

<figure style="text-align: center;" id="fig-2-3">
    <img src="images/fig-2.gif" width="70%" alt="Fig. 2" style="display: inline-block;">
    <figcaption align="center"></figcaption>
    <img src="images/fig-3.gif" width="70%" alt="Fig. 3" style="display: inline-block;">
    <figcaption align="center"></figcaption>
</figure>

## Citation

Cited as:

    Son, Sunny. (Aug 2023). "Connecting the Dots (or Words)". Sunny's Notes. 
    https://sunnyson.dev/notes/2023/08/connecting-the-dots-or-words/.

Or, in BibTeX format:

<pre tabindex="0"><code>@article{son-2023-monocle-pt1,
  title   = &quot;Connecting the Dots (or Words)&quot;,
  author  = &quot;Son, Sunny&quot;,
  journal = &quot;sunnyson.dev&quot;,
  year    = &quot;2023&quot;,
  month   = &quot;August&quot;,
  url     = &quot;https://sunnyson.dev/notes/2023/08/connecting-the-dots-or-words/&quot;
}
</code></pre>

## References

[1] Trapnell, C. et al. <a href="https://www.nature.com/articles/nbt.2859">&ldquo;The dynamics and regulators of cell fate decisions are revealed by pseudotemporal ordering of single cells.&quot;</a> Nature Biotechnology 2014

[2] Qiu, Xiaojie. et al. <a href="https://www.nature.com/articles/nmeth.4402">&ldquo;Reversed graph embedding resolves complex single-cell trajectories&quot;</a> Nature Methods 2017