[← Research](/research) · Nov 20, 2026

# Interpreting Inkling 2 Mini

We studied how this looped model propagates information across blocks and loops. Decomposing the inputs at each layer allows us to easily understand which layers, blocks, and loops communicate with one another.

### Authors

Andrew Lee

Kamesh Krishnamurthy

Jack Merullo

Dan Balsam

Ekdeep Singh Lubana

### Published

November 20, 2026

Contents

Contents ![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0idG9jLWNoZXZyb24iIHZpZXdib3g9IjAgMCAzMiAzMiIgYXJpYS1oaWRkZW49InRydWUiIGRhdGEtYXN0cm8tY2lkLW5jbjdkeTNtPjxwYXRoIGQ9Ik0yMiAxNiAxMiAyNmwtMS40LTEuNCA4LjYtOC42LTguNi04LjZMMTIgNnoiIGZpbGw9ImN1cnJlbnRDb2xvciIgZGF0YS1hc3Ryby1jaWQtbmNuN2R5M20gLz48L3N2Zz4=)

- [Overview of Inkling 2 Mini’s architecture](#overview-of-inkling-2-minis-architecture)

- [Understanding information flow in Inkling 2 Mini: decomposing input components](#understanding-information-flow-in-inkling-2-mini-decomposing-input-components)

- [Coordination across layers, blocks, and loops](#coordination-across-layers-blocks-and-loops)
  ![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0idG9jLWNoZXZyb24iIHZpZXdib3g9IjAgMCAzMiAzMiIgYXJpYS1oaWRkZW49InRydWUiIGRhdGEtYXN0cm8tY2lkLW5jbjdkeTNtPjxwYXRoIGQ9Ik0yMiAxNiAxMiAyNmwtMS40LTEuNCA4LjYtOC42LTguNi04LjZMMTIgNnoiIGZpbGw9ImN1cnJlbnRDb2xvciIgZGF0YS1hc3Ryby1jaWQtbmNuN2R5M20gLz48L3N2Zz4=)

  - [Weight space analysis of block-wise coordination](#weight-space-analysis-of-block-wise-coordination)
  - [Activation space analysis of loop-wise coordination (+ massive outlier activations)](#activation-space-analysis-of-loop-wise-coordination--massive-outlier-activations)

- [Decoding loops: logit-lens, tuned-lens, coda-lens](#decoding-loops-logit-lens-tuned-lens-coda-lens)

- [Conclusion](#conclusion)

- [Appendix: detailed architecture of Inkling 2 Mini](#architecture-details)

Today, Thinking Machines [released Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini), a small open-weights model. The model combines multiple architectural innovations, including attention residuals ^([1](#user-content-fn-attnres)) and looping its middle layers ^([2](#user-content-fn-looping)).

Thinking Machines provided Goodfire with early access to the weights of Inkling 2 Mini so we could investigate how these architectural innovations affect the model’s internal mechanisms. We studied a few key questions:

- How does information flow through the model?
- How do model components coordinate that information flow across layers and loops?
- How can we decode such information that is being processed throughout the network?

Our preliminary findings include insights into how layers coordinate with each other, and with future and past versions of themselves: counterintuitively, looped model layers can write to themselves, or read from “future” layers!

We believe these insights are just the beginning of what we can learn about these models. As their architectures continue to evolve, we believe our intuitions and understanding of their internal mechanisms will be able to evolve with them; we hope that interpreting Inkling 2 Mini will contribute to the cross-pollination of insights across different kinds of models.

## Overview of Inkling 2 Mini’s architecture

Here we provide a high-level overview of Inkling 2 Mini’s architecture. A detailed description can be found in Thinking Machines' [release announcement](https://thinkingmachines.ai/news/inkling-2-mini), or our [Appendix](#architecture-details).

Inkling 2 Mini has 24 layers, each layer consisting of an attention sublayer followed by a mixture-of-experts (MoE) sublayer. The model has two major architectural differences from a vanilla transformer.

First, Inkling 2 Mini uses *attention residuals*^([1](#user-content-fn-attnres)): each layer reads from previous layer outputs using attention, rather than reading from a cumulative residual stream. More specifically, it uses depth-wise attention over *blocks*, each of which sums the output of 6 layers. The model's 24 layers are divided into 4 blocks. That means there is no single residual stream running through the network. Instead, each block keeps a running sum of its own layers’ outputs, starting from zero. Each block eventually outputs the accumulated sum: see [Figure 1](#figure-1).

![Figure 1: Depth-wise attention attends over block outputs to compute the input to each layer. This diagram represents model computation midway through the first loop, so omits the additional loops and the fourth block.](/_astro/depthwise_attention_blocks.DkGl-xKN.jpg)

Figure 1: Depth-wise attention attends over block outputs to compute the input to each layer. This diagram represents model computation midway through the first loop, so omits the additional loops and the fourth block.

Second, Inkling 2 Mini is a looped model: before being decoded, the model’s intermediate computations go through the same set of weights multiple times. Not all the layers are looped, however: the first block (layers 1-6) runs once on the input, followed by the two core blocks (layers 7-18), *which are looped three times*. The final block (block 4, layers 19-24) then runs once to decode the output.

The cross-loop attention allows information to pass across loops. In particular, during the second or third pass of the core blocks, each sublayer uses its cross-loop attention to read from the output of blocks 2 and 3 from the previous loop: see [Figure 2](#figure-2).

![Figure 2: Each layer can read information from all earlier blocks, as well as the summed outputs of the earlier layers in its block (depth-wise attention; blue arrows). Layers in blocks 2 and 3, which are looped, can also read from the previous loop (cross-loop attention; pink arrows). All of this is over a single token; standard sequence attention is used by the attention sublayers within each layer.](/_astro/layer_inputs_blocks.Bnj1US5P.jpg)

Figure 2: Each layer can read information from all earlier blocks, as well as the summed outputs of the earlier layers in its block (depth-wise attention; blue arrows). Layers in blocks 2 and 3, which are looped, can also read from the previous loop (cross-loop attention; pink arrows). All of this is over a single token; standard sequence attention is used by the attention sublayers within each layer.

Together, that creates two channels for information to flow across blocks: via depth-wise attention and cross-loop attention. For each of these channels, each sublayer has a dedicated attention module to read from previous blocks or previous loops.

![Figure 3: Overview of Inkling 2 Mini’s architecture.](/_astro/three_loop_architecture.BIXNAxOD.png)

Figure 3: Overview of Inkling 2 Mini’s architecture.

Put together, the entire architecture (and flow of information) looks like [Figure 3](#figure-3). For a more detailed description, see our [Appendix](#architecture-details). Conveniently, Inkling 2 Mini’s architecture allows us to decompose the input of each layer as a weighted sum of the outputs of previous layers, allowing us to easily visualize and understand how information travels across the network. We describe this decomposition in the next section.

## Understanding information flow in Inkling 2 Mini: decomposing input components

![Figure 4: Information flow in Inkling 2 Mini. The input to each layer can be decomposed as the weighted sum of earlier layer outputs, allowing us to visualize how information flows throughout the network.](/_astro/cross_block_reads_row_inputs_coda_at_row_labeled.DsNS7xSJ.gif)

Figure 4: Information flow in Inkling 2 Mini. The input to each layer can be decomposed as the weighted sum of earlier layer outputs, allowing us to visualize how information flows throughout the network.

Here we provide a mental model of how information propagates through Inkling 2 Mini.

At a high level, each layer’s input (denoted $`\mathbf{x}`$) is a mixture of the current or earlier blocks, which themselves are also each the sum of outputs coming from its layers. This allows us to decompose the input at every layer into a sum of contributions from earlier layers, allowing us to track how information flows from one layer to another, across blocks or even loops.

In Inkling 2 Mini, the input to each layer is a softmax mix over the current or earlier blocks, which is built via the model’s depth-wise attention: $`\mathbf{x}=\sum_i\alpha_ir_i\mathbf{v}_i`$, where $`\mathbf{v}_i`$ is the output of block $`i`$, $`r_i`$ is an RMS-normalization term ($`\frac{1}{RMS(\mathbf{v}_i)}`$), and $`\alpha_i`$ is the attention weight applied to block $`i`$.

The output of each block is the sum of its layers’ outputs: $`\mathbf{v}_i = \sum_{w\in B_i}\mathbf{o}_w`$, where $`B_i`$ refers to block $`i`$ and $`\mathbf{o}_w`$ is the output of a specific attention or MoE layer from block $`i`$. This allows us to decompose $`\mathbf{x}`$ further: $`\mathbf{x} = \sum_i\alpha_ir_i\mathbf{v}_i = \sum_w\alpha_{i(w)}r_{i(w)}\mathbf{o}_w`$, where $`\alpha_{i(w)}`$, $`r_{i(w)}`$ are the attention weight and RMS-normalization term for block $`i`$ that the output $`\mathbf{o}_w`$ belongs to.

The next stage mixes a message $`\mathbf{m}`$ from the previous loop. Let $`t`$ index the current loop’s turn, while $`t-1`$ denotes the previous loop. The message is also a softmax mix of blocks 2 and 3 from the previous loop: $`\mathbf{m}=\beta_2r_2^{[t-1]}\mathbf{v}_2^{[t-1]} + \beta_3r_3^{[t-1]}\mathbf{v}_3^{[t-1]}`$. Here, $`\beta_{2, 3}`$ are attention weights on blocks 2 and 3, $`r_{2,3}^{[t-1]}`$ refer to the RMS-normalizer for blocks 2 and 3 from the previous loop, and $`\mathbf{v}_{2, 3}^{[t-1]}`$ are the outputs of blocks 2 and 3 from the previous loop.

The message $`\mathbf{m}`$ is mixed into the current layer’s input $`\mathbf{x}`$ via a “keep” term ($`\boldsymbol{\kappa}`$) and “gain” term ($`\boldsymbol{\gamma}`$): $`\mathbf{x}'=\boldsymbol{\kappa}\odot  \mathbf{x} + \boldsymbol{\gamma}\odot\mathbf{m}`$ (see [Appendix](#architecture-details) for how $`\boldsymbol{\kappa}`$ and $`\boldsymbol{\gamma}`$ are computed). Note that the first term, $`\boldsymbol{\kappa}\odot\mathbf{x}`$, comes from the current loop, while the second term, $`\boldsymbol{\gamma}\odot\mathbf{m}`$, comes from the previous loop. This allows us to rewrite $`\mathbf{x'}`$ as the sum of previous layers: $`\mathbf{x'}=\sum_w\mathbf{c}_w`$, where

- $`\mathbf{c}_w = \boldsymbol{\kappa}\odot\alpha_{i(w)} r_{i(w)}^{[t]} \mathbf{o}_w`$ for all writes from the current loop.
- $`\mathbf{c}_w =  \boldsymbol{\gamma}\odot\beta_{i(w)} r_{i(w)}^{[t-1]} \mathbf{o}_w`$ for all writes from the previous loop.

Thus we have decomposed $`\mathbf{x'}`$ into a weighted sum of layer outputs $`\mathbf{o}_w`$. Finally, we can attribute to each $`\mathbf{o}_w`$ its share of the input: $`s_w = \frac{\mathbf{c}_w \cdot \mathbf{x}'}{\|\mathbf{x}'\|^2}`$, where $`s_w`$ is the coefficient of $`\mathbf{c}_w`$’s projection onto $`\mathbf{x}'`$. Because the terms sum to the input, the shares sum to exactly 1.

[Figure 4](#figure-4) above shows how information flows within Inkling 2 Mini according to our decomposition. Per layer, we take the average contributions over 13,233 tokens coming from 400 FineWeb documents. This demonstrates the direct contributions from one layer to another.

There are a few observations to make. Regarding depth-wise attention, the model attends more heavily towards recent layers, akin to a sliding window. Regarding cross-loop attention, later loops tend to attend to Block 3 over Block 2. This is especially prominent in Block 3, where it almost always attends to itself from its previous loop, perhaps because it already has access to information from Block 2 via the depth-wise attention. With these intuitions in mind, we next study some of the mechanisms underlying these information flows.

## Coordination across layers, blocks, and loops

Here we study how Inkling 2 Mini coordinates the flow of information across blocks and loops, both in weight space and activation space.

### Weight space analysis of block-wise coordination

Note that both depth-wise and cross-loop attention read at the block level. These two attention mechanisms come with their own set of weights that are used to determine which blocks to attend to. We refer to these weights as “reader” weights. For depth-wise attention, the model has a total of 56 readers (31 for attention sublayers, 24 for MoE sublayers, and 1 before the final output layer), while for cross-loop attention, the 12 layers in blocks 2 and 3 (the looped layers) each have 4 readers ($`\mathbf{u}_{Q,K,V,MoE}`$), for a total of 48. As such, we have a total of 104 reader weights facilitating depth-wise attention and cross-loop attention.

In short, we find that late layers of each block, in particular the last layers, write heavily for these readers, presumably “tagging” the activations such that later layers know which blocks to read from.

To understand which components “write” towards these readers, we inspect every single down-projection column vector in the MoE weights of all layers. More specifically, each MoE has 147,456 neurons, and each neuron has an associated weight vector that gets added to the model’s activation when the neuron is active.

Thus we have 24 layers x 147,456 = 3,538,944 weight vectors that can write into the model’s activations. Which of these are writing for any of the 104 attention readers?

To answer this, we look at all possible pairs between the 3.5M writers (denoted $`\mathbf{w}`$’s for “writers”) and 104 readers (denoted $`\mathbf{r}`$’s for “readers”), for a total of 368 million pairs. We measure the absolute cosine similarity between every pair of ($`\mathbf{w}`$, $`\mathbf{r}`$) vectors, and look at the top 0.001% pairs, which yields a threshold absolute cosine similarity value of 0.30. By considering the pairs of writers and readers that exceed this threshold, we have 3,490 pairs remaining.

[Figures 5](#figure-5) and [6](#figure-6) below break these 3,490 pairs down by the writer’s layer (x-axis) and the reader’s layer (bar stacks, where the colors indicate the reader’s layer). The y-axis indicates the number of (writer, reader) pairs whose absolute cosine similarity exceeds 0.30. We indicate the first layer of each block with stripes in its bar stack. [Figure 5](#figure-5) demonstrates depth-wise attention writer-reader pairs, while [Figure 6](#figure-6) demonstrates cross-loop attention pairs.

![Figure 5: Writer, reader pairs with highest alignment for depth-wise attention. The writer-reader pairs with high alignment are concentrated in the last layers of each block, which we interpret as these neurons “tagging” the activations for later layers to use in order to read from its corresponding block.](/_astro/pairs_by_reader_depthmix_30.ebNV-buR.jpg)

Figure 5: Writer, reader pairs with highest alignment for depth-wise attention. The writer-reader pairs with high alignment are concentrated in the last layers of each block, which we interpret as these neurons “tagging” the activations for later layers to use in order to read from its corresponding block.

![Figure 6: Writer, reader pairs with highest alignment for cross-loop attention. The writer-reader pairs with high alignment are concentrated in the last layer of block 3, such that future loops read from block 3. Indeed, Inkling 2 Mini almost always reads from block 3 in loops 2 and 3 (see Figure 4).](/_astro/pairs_by_reader_crossloop_30.CTGHi8-a.jpg)

Figure 6: Writer, reader pairs with highest alignment for cross-loop attention. The writer-reader pairs with high alignment are concentrated in the last layer of block 3, such that future loops read from block 3. Indeed, Inkling 2 Mini almost always reads from block 3 in loops 2 and 3 (see [Figure 4](#figure-4)).

There are a few observations to note. For the depth-wise reader-writer interactions, you can see most of the writing taking place in the last layer of each block (layers 6, 12, 18). For instance, layer 6 writes mostly to block 4, layer 12 writes mostly to blocks 3 and 4, and lastly layer 18 writes to block 4. One can interpret these interactions as the final layers of each block “tagging” the activations for future layers to use, such that later layers can read from these preceding blocks.

The cross-loop interactions tell a similar story, in that all layers (but especially the final layer) of Block 3 do the majority of the writing, in particular to layer 13 (first layer of block 3). Similar to the depth-wise case, this allows layer 13 in future loops to read from Block 3, which is indeed the dynamic that we observe in our previous analysis (see [Figure 4](#figure-4)).

To summarize, late layers in each block seem to “coordinate” with other layers in order to facilitate the information flow into downstream layers or future loops, whether it is via depth-wise or cross-loop attention.

### Activation space analysis of loop-wise coordination (+ massive outlier activations)

These cross-loop coordinations can be inspected from the activations of the model as well. In particular, we find massive outlier activations that seem to determine which blocks the model should attend to across loops.

#### Massive outlier activations

Our analysis starts with a slight detour, by introducing some massive outlier coordinate values in the model’s activations. In particular, we focus on the massive activations that occur in the looped layers (blocks 2, 3).

To identify outliers, we take the activations across all 24 layers and normalize them based on their RMS, motivated by the architecture’s extensive use of RMSNorm. Each normalized vector therefore has a squared norm of d = 2304. If the squared norm were distributed equally across all coordinates, we would expect each coordinate to have a magnitude of 1, contributing approximately 0.043% of the total squared norm.

Surprisingly, we observe cases where a single coordinate has an average normalized value of +32.4, occupying $`32.4^2/2304 \approx 46\%`$ of the squared norm.

Although these outlier activations might serve several functions, our findings suggest that they play a role in facilitating information flow between loops.

One example of an outlier is coordinate 2027 (out of 2304 coordinates), which we denote $`d_{2027}`$. As alluded to earlier, a single coordinate $`d_{2027}`$ has an average RMS-normalized value of +32.4, occupying 46% of the activation’s total squared norm. This coordinate has the most anomalous magnitude, and thus we focus on this specific coordinate for now.

#### Writing onto $`d_{2027}`$ near the end of a loop

The first hint for the role of $`d_{2027}`$ comes from inspecting when its value surges. Given a small set of 13,233 tokens from 400 FineWeb documents, we observe that $`d_{2027}`$ surges at layer 18, the end of Block 3 of the first two loops: see [Figure 7](#figure-7).

As we will see in the next analysis, we believe this is done in order to prepare future loops to read from Block 3.

![Figure 7: Components writing to d₂₀₂₇. We believe d₂₀₂₇ is used across loops to coordinate which blocks the future loops should read from. Note that for layer 18 (last layer of each loop), the first two loops have a massive positive spike, telling future loops to read from Block 3 (Note that indeed, loops 2 and 3 always read from block 3 — see Figure 4). On the other hand, in loop 3, layer 18 has a large negative spike, presumably because the model does not expect any future loops.](/_astro/coord2027_writers_blog.CjyUw31h.jpg)

Figure 7: Components writing to *d*₂₀₂₇. We believe *d*₂₀₂₇ is used across loops to coordinate which blocks the future loops should read from. Note that for layer 18 (last layer of each loop), the first two loops have a massive positive spike, telling future loops to read from Block 3 (Note that indeed, loops 2 and 3 always read from block 3 — see [Figure 4](#figure-4)). On the other hand, in loop 3, layer 18 has a large negative spike, presumably because the model does not expect any future loops.

#### Reading from $`d_{2027}`$ in later loops

Taking a step back from outlier activations, let us revisit the attention dynamics across loops ([Figure 8](#figure-8)). In particular, consider the beginning of each block in the second and third loops. In all cases, these layers strongly attend to block 3 from the preceding loop (orange curves are much thicker than the green curves).

![Figure 8: Attention patterns over blocks of earlier loops. Orange and green curves indicate which blocks each layer attends to.](/_astro/information_flow_panels_row_inputs.DcfZ6Fta.jpg)

Figure 8: Attention patterns over blocks of earlier loops. Orange and green curves indicate which blocks each layer attends to.

To understand why, let us revisit Inkling 2 Mini’s architecture. Each layer has dedicated weight vectors, $`\mathbf{u}_Q, \mathbf{u}_V, \mathbf{u}_K, \mathbf{u}_{MoE} \in \mathbb{R}^d`$, to determine which block from its preceding loop to read from. For now, let us only consider $`\mathbf{u}_Q`$ (the same analysis applies to all 4 weights):

$`\hat{\mathbf{v}}_i = \operatorname{RMSNorm}(\mathbf{v}_i),      \mathbf{v}_i \in \mathbb{R}^d`$: Output of block $`i`$ from the previous loop.\
$`s_i = \frac{\mathbf{u}_Q \cdot \hat{\mathbf{v}}_i}{\sqrt{d}}`$: Score assigned to block $`i`$.\
$`\beta_i = \operatorname{softmax}([s_1, \dots, s_N])`$: Attention weights over blocks.

The attention values $`\beta`$ determine which of the previous blocks to attend to. In our case, we only have two blocks, and we denote their corresponding attention weights as $`\beta_2`$ and $`\beta_3`$ (for blocks 2 and 3).

Again, we often see the model attend mostly to block 3, which we can spell out as follows:

$`\beta_3 = \frac{e^{s_3}}{e^{s_2} + e^{s_3}}            = \frac{1}{1 + e^{-(s_3 - s_2)}}            = \sigma(s_3 - s_2)`$,

$`s_3 - s_2 = \frac{\mathbf{u}_Q \cdot \left(\hat{\mathbf{v}}_3 - \hat{\mathbf{v}}_2\right)}{\sqrt{d}} =   \frac{\mathbf{u}_Q \cdot \mathbf{\delta}}{\sqrt{d}}`$.

So ultimately whether the model attends to block 2 versus block 3 depends on the value $`\mathbf{u}\cdot\mathbf{\delta}`$, which we can decompose coordinate-wise: $`\text{gap} = \mathbf{u}\cdot\mathbf{\delta}=\sum_i\mathbf{u}_i\mathbf{\delta}_i`$. With this element-wise decomposition, we can finally inspect which coordinate $`i`$ contributes most towards $`\mathbf{u}\cdot\mathbf{\delta}`$.

To our surprise, this is where our previous coordinate of interest, $`d_{2027}`$, returns. In the case of layer 13 (note that this is the first layer of block 3) during the second loop, for its 4 cross-loop readers $`\mathbf{u}_Q, \mathbf{u}_V, \mathbf{u}_K, \mathbf{u}_{MoE}`$, we see $`d_{2027}`$ being the largest (or 2nd largest) contributor towards each of their respective gap terms, contributing 17*%*, 19%, 11%, and 6% respectively on its own!

Note that $`d_{2027}`$ is not the only outlier that contributes heavily towards layer 13’s cross-loop readers. The top 5 activation coordinates that contribute the most towards $`\mathbf{u}_{Q, V, K, MoE}`$’s gap terms account for 60%, 62%, 32%, and 30% respectively. Interestingly, these 5 coordinates are also within the top 15 coordinates when ranked by their RMS-normalized norms.

Here let us briefly return to [Figure 7](#figure-7). Note that at the end of the loops (Loop 3, layer 18), there are no more loops to follow, and thus the MoE and attention heads at that layer “clean up” the heavy activation on $`d_{2027}`$, before moving on to the final block. Our speculation is that because cross-loop communications are no longer needed, the model removes the activity on $`d_{2027}`$.

## Decoding loops: logit-lens, tuned-lens, coda-lens

While the previous analyses study how information flows throughout the network, here we analyze how to decode this information. One commonly used trick is to decode intermediate layers by applying the unembedding layer at intermediate hidden states, often referred to as logit-lens. How well does logit-lens (and variants) work with Inkling 2 Mini’s loops?

Here we study a few variants of logit-lens:

- **Logit-lens**: Apply the final decoding head at intermediate hidden-states.
- **Tuned-lens**: Learn an affine transformation from a hidden-state to the model’s final layer, in order to preserve the model’s output distribution.
- **J-lens:** Use an averaged Jacobian matrix from hidden-states to the model’s final state.
- **Coda-lens:** Here we refer to the final block (block 4) as the coda, and apply it on the model’s hidden-states (as opposed to applying just the final unembedding layer). Note that unlike the above methods, coda-lens is applied to the entire context’s hidden-states, as the coda itself has sequential attention heads in its layers.

We evaluate how well each variant works with two metrics: 1) How often does each lens’ top 50 tokens include the model’s final output token? and 2) How much probability mass does each lens put on the model’s top 50 tokens?^([3](#user-content-fn-lens)) We evaluate each method using 1,250 FineWeb documents. Tuned-lens requires training, for which we use 23,750 FineWeb documents.

[Figure 9](#figure-9) shows the results. In short, we find that logit lens does not work well, apart from the final few layers per loop, which is where the coda block is eventually applied, while tuned-lens and coda-lens work very well.

Also note that logit-lens typically only works on mid to late layers for standard vanilla transformers, closer to where the unembedding is expected to be applied. In the case of looped models, despite the layers from early loops being “early” in terms of the forward pass, they seem to be much more interpretable, while the mid-layers per block across all loops seem to be less so.

This is great news for interpretability, as it suggests that even with these architectural modifications, the model’s internal computations are still interpretable!

![Figure 9. While logit-lens does not work well in decoding earlier layers, tuned-lens and coda-lens work well.](/_astro/lens_grid_heldout_looped.CkiV4VHS.jpg)

Figure 9. While logit-lens does not work well in decoding earlier layers, tuned-lens and coda-lens work well.

## Conclusion

These are early results, and there is much more science to be done - including as part of [NDIF’s Looped Lens Challenge](http://ndif.us/looped-lens-challenge). We conclude with a note of excitement: as neural network architectures continue to evolve, we believe our intuitions and understanding of their internal mechanisms will be able to evolve with them. Just as interpretability tools and ideas have generalized across [CNNs](https://distill.pub/2020/circuits/zoom-in/), [transformers](https://transformer-circuits.pub/2021/framework/index.html), and [hybrid architectures](/research/interpreting-evo-2), we hope that interpreting Inkling 2 Mini will contribute to the cross-pollination of insights across different kinds of models.

### Citation

Andrew Lee, Kamesh Krishnamurthy, Jack Merullo, Dan Balsam, and Ekdeep Singh Lubana. “Interpreting Inkling 2 Mini.” Goodfire, November 20, 2026.

BibTeX:

Copy

```
@misc{lee2026inkling,
  author = {Andrew Lee and Kamesh Krishnamurthy and Jack Merullo and Dan Balsam and Ekdeep Singh Lubana},
  title = {Interpreting {Inkling 2 Mini}},
  year = {2026},
  publisher = {Goodfire},
  url = {https://www.goodfire.com/research/interpreting-inkling-2-mini/}
}
```

## Footnotes

1.  [Kimi Team](https://arxiv.org/pdf/2603.15031) [↩](#user-content-fnref-attnres) [↩²](#user-content-fnref-attnres-2) [↩³](#user-content-fnref-attnres-3)

2.  [Dehghani et al.](https://arxiv.org/pdf/1807.03819), [Yang et al.](https://arxiv.org/pdf/2311.12424), [Geiping et al.](https://arxiv.org/pdf/2502.05171), [Bae et al.](https://arxiv.org/pdf/2507.10524), [Nanbeige LLM Lab](https://arxiv.org/pdf/2607.22083), [Zhu et al.](https://arxiv.org/pdf/2510.25741) [↩](#user-content-fnref-looping)

3.  Note that these metrics are likely not suitable for Jacobian-lens, as these metrics do not reflect the objective of J-lens very well. We leave the exploration of J-lens on looped models for future work. [↩](#user-content-fnref-lens)

## Appendix: detailed architecture of Inkling 2 Mini

Here we provide an overview of Inkling 2 Mini’s architecture. Compared to a vanilla transformer, there are two main differences to consider: attention residuals ^([1](#user-content-fn-attnres)) and looped blocks.

**Notation:**

- $`d = 2304`$: model width. $`\ell \in \{1,\dots,24\}`$: layer. $`\mathcal{B}_i`$: the layers of block $`i`$.
- $`\mathbf{o}^{(\ell)}_{\text{attn}}, \mathbf{o}^{(\ell)}_{\text{moe}} \in \mathbb{R}^d`$: Outputs of layer $`\ell`$'s attention and MoE sublayers.
- $`\mathbf{v}_i \in \mathbb{R}^d`$: Sum of block $`i`$.
  - $`\mathbf{v}_0`$: The token embeddings
  - $`\mathbf{v}_i^{(\ell)}`$: The running sum within block $`i`$, just before layer $`\ell`$
  - $`\mathbf{v}_i^{[t]}`$: The sum of block $`i`$ in loop $`t`$.
  - Note that $`\mathbf{v}_i`$ is the sum of the outputs $`w`$ in block $`i`$: $`\mathbf{v}_i=\sum_{w \in i} \mathbf{o}_w`$
- $`\hat{\mathbf{v}} = \mathbf{v}/\mathrm{RMS}(\mathbf{v})`$.
- $`\mathbf{q}`$, $`\mathbf{u} \in \mathbb{R}^d`$: learned queries for depth-wise and cross-loop attention.
- $`\mathbf{m} \in \mathbb{R}^d`$: cross-loop message; $`\boldsymbol{\kappa}, \boldsymbol{\gamma}`$: keep and gain.

**Layers.** Inkling 2 Mini has 24 layers of width $`d = 2304`$, each consisting of an attention sublayer followed by a mixture-of-experts (MoE) sublayer.

**Blocks.** The layers are grouped into 4 blocks of 6 consecutive layers: $`\mathcal{B}_1 = \{1,\dots,6\}`$, $`\mathcal{B}_2 = \{7,\dots,12\}`$, $`\mathcal{B}_3 = \{13,\dots,18\}`$, $`\mathcal{B}_4 = \{19,\dots,24\}`$.

Unlike vanilla transformers, there is no single residual stream running through the network. Instead, each block keeps a running sum of its own layers’ outputs, starting from zero. The running sum of block $`i`$ just before layer $`\ell \in \mathcal{B}_i`$ is

``` math
\mathbf{v}_i^{(\ell)} = \sum_{\ell' \in \mathcal{B}_i,\ \ell' < \ell} \big(\mathbf{o}_{attn}^{(\ell')} + \mathbf{o}_{moe}^{(\ell')}\big),
```

and $`\mathbf{v}_i`$ is the sum over all six layers. The token embedding acts as an extra "Block 0", $`\mathbf{v}_{0} = \mathrm{RMSNorm}(\mathbf{e}_x)`$, where $`\mathbf{e}_x`$ is the token embedding for token $`x`$.

**Depth-wise attention.** Unlike vanilla transformers, in which each sublayer reads from the residual stream, each sublayer of Inkling 2 Mini reads its input by taking a softmax mix over the block sums available at that point: the embeddings $`\mathbf{v}_0`$, completed blocks $`\mathbf{v}_1,\dots,\mathbf{v}_{i-1}`$, and the current running sum ($`\mathbf{v}_i^{(\ell)}`$ for attention, $`\mathbf{v}_i^{(\ell)} + \mathbf{o}^{(\ell)}_{attn}`$ for the MoE). For some layer $`\ell`$ with learned query $`\mathbf{q}`$,

``` math
\mathbf{x} = \sum_j \alpha_{j}\,\hat{\mathbf{v}}_j, \qquad
\alpha_{j} = \operatorname{softmax}_j\!\left(\frac{\mathbf{q}^\top \hat{\mathbf{v}}_j}{\sqrt{d}}\right), \qquad
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\mathrm{RMS}(\mathbf{v})}.
```

and the sublayer receives $`\mathrm{RMSNorm}(\mathbf{x})`$. Put differently, the input $`\mathbf{x}`$ is a weighted sum of the earlier blocks. However, the input $`\mathbf{x}`$ is first “mixed” with messages coming from previous loops, as explained below.

**Loops.** Blocks 2 and 3 (layers 7–18) form the loop and run $`T = 3`$ times, while the first and last blocks run once. At each loop $`t`$, the depth-wise sources that the depth-wise attention reads from include $`\mathbf{v}_0`$ (the token embeddings), $`\mathbf{v}_1`$, and the current loop $`t`$’s own sums $`\mathbf{v}_2^{[t]}, \mathbf{v}_3^{[t]}`$. At the end of 3 loops, the final block reads the last loop’s $`\mathbf{v}^{[3]}_2, \mathbf{v}^{[3]}_3`$.

**Cross-loop attention.** In loops $`t \ge 2`$, each looped layer $`\ell`$ has four inputs $`s \in \{Q, K, V, \mathrm{MoE}\}`$, which each reads a message from the previous loop's two block sums using its own learned query $`\mathbf{u}_{s}`$:

``` math
\boldsymbol{\mathbf{m}}_{s} = \sum_{j \in \{2,3\}} \beta_j\,\hat{\mathbf{v}}_j^{[t-1]}, \qquad
\beta_j = \operatorname{softmax}_j\!\left(\frac{\mathbf{u}_{s}^\top \hat{\mathbf{v}}_j^{[t-1]}}{\sqrt{d}}\right)
```

**Mixing the cross-loop messages in.** Before an input vector $`\mathbf{x}`$ enters a sublayer, the message $`\mathbf{m}`$ from the previous loop is mixed into the input, by applying a “keep” term ($`\boldsymbol{\kappa}`$) and “gain” term ($`\boldsymbol{\gamma}`$) on the input and message respectively:

``` math
\begin{aligned}
\mathbf{z} &= \mathbf{P}\,\big[\mathrm{RMSNorm}(\mathbf{x});\ \mathrm{RMSNorm}(\mathbf{m})\big], & \mathbf{P} &\in \mathbb{R}^{64 \times 2d},\\
\boldsymbol{\kappa} &= 1 - \tanh(\mathbf{F}\mathbf{z}), \quad \boldsymbol{\gamma} = \tanh(\mathbf{G}\mathbf{z}), & \mathbf{F}, \mathbf{G} &\in \mathbb{R}^{d \times 64},\\
\mathbf{x}' &= \boldsymbol{\kappa} \odot \mathbf{x} + \boldsymbol{\gamma} \odot \mathbf{m}.
\end{aligned}
```

The keep term $`\boldsymbol{\kappa} \in (0, 2)`$ either dampens or amplifies the current loop’s input, while the gain term $`\boldsymbol{\gamma} \in (-1, 1)`$ either adds or subtracts the message.

**Output.** The final output is a depth-wise mix over $`\mathbf{v}_{0, \dots, 4}`$, followed by an RMSNorm and the unembedding.

## Research

[](/research/neuron-communities)

### Discovering Neuron Communities in Language Models

November 17, 2026

Igor Shilov, Thomas Fel, Tal Haklay, Lucius Bushnaq, Owen Lewis

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjYgMjYiIGZpbGw9Im5vbmUiPjxjaXJjbGUgY3g9IjEzIiBjeT0iMTMiIHI9IjEyIiBmaWxsPSJibGFjayI+PC9jaXJjbGU+PHBhdGggZD0iTTcgMTMuODU5Mkw3IDEyLjIwNjJMMTYuMDg4NSAxMi4yMDYyTDEzLjYyOTUgOC44Njc0M0wxNS4zNjA3IDhMMTkgMTMuMDA4MkwxNS4zODAzIDE4TDEzLjYyOTUgMTcuMTMyNkwxNi4wODg1IDEzLjg1OTJMNyAxMy44NTkyWiIgZmlsbD0id2hpdGUiIC8+PC9zdmc+) ![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iYXJyb3ctaG92ZXIiIHZpZXdib3g9IjAgMCAyNiAyNiIgZmlsbD0ibm9uZSI+PGNpcmNsZSBjeD0iMTMiIGN5PSIxMyIgcj0iMTIiIGZpbGw9IiNGRkVCQUEiPjwvY2lyY2xlPjxwYXRoIGQ9Ik03IDEzLjg1OTJMNyAxMi4yMDYyTDE2LjA4ODUgMTIuMjA2MkwxMy42Mjk1IDguODY3NDNMMTUuMzYwNyA4TDE5IDEzLjAwODJMMTUuMzgwMyAxOEwxMy42Mjk1IDE3LjEzMjZMMTYuMDg4NSAxMy44NTkyTDcgMTMuODU5MloiIGZpbGw9ImJsYWNrIiAvPjwvc3ZnPg==)

[](/research/neurons-as-an-interpretable-basis)

### Neurons as an Interpretable Basis

November 10, 2026

Curt Tigges, Matthew Kowal, Thomas McGrath, Owen Lewis, Nick Cammarata

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjYgMjYiIGZpbGw9Im5vbmUiPjxjaXJjbGUgY3g9IjEzIiBjeT0iMTMiIHI9IjEyIiBmaWxsPSJibGFjayI+PC9jaXJjbGU+PHBhdGggZD0iTTcgMTMuODU5Mkw3IDEyLjIwNjJMMTYuMDg4NSAxMi4yMDYyTDEzLjYyOTUgOC44Njc0M0wxNS4zNjA3IDhMMTkgMTMuMDA4MkwxNS4zODAzIDE4TDEzLjYyOTUgMTcuMTMyNkwxNi4wODg1IDEzLjg1OTJMNyAxMy44NTkyWiIgZmlsbD0id2hpdGUiIC8+PC9zdmc+) ![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iYXJyb3ctaG92ZXIiIHZpZXdib3g9IjAgMCAyNiAyNiIgZmlsbD0ibm9uZSI+PGNpcmNsZSBjeD0iMTMiIGN5PSIxMyIgcj0iMTIiIGZpbGw9IiNGRkVCQUEiPjwvY2lyY2xlPjxwYXRoIGQ9Ik03IDEzLjg1OTJMNyAxMi4yMDYyTDE2LjA4ODUgMTIuMjA2MkwxMy42Mjk1IDguODY3NDNMMTUuMzYwNyA4TDE5IDEzLjAwODJMMTUuMzgwMyAxOEwxMy42Mjk1IDE3LjEzMjZMMTYuMDg4NSAxMy44NTkyTDcgMTMuODU5MloiIGZpbGw9ImJsYWNrIiAvPjwvc3ZnPg==)

[](/research/public_neuron_atlas/all_labels)

### Neuron Atlas

November 10, 2026

![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjYgMjYiIGZpbGw9Im5vbmUiPjxjaXJjbGUgY3g9IjEzIiBjeT0iMTMiIHI9IjEyIiBmaWxsPSJibGFjayI+PC9jaXJjbGU+PHBhdGggZD0iTTcgMTMuODU5Mkw3IDEyLjIwNjJMMTYuMDg4NSAxMi4yMDYyTDEzLjYyOTUgOC44Njc0M0wxNS4zNjA3IDhMMTkgMTMuMDA4MkwxNS4zODAzIDE4TDEzLjYyOTUgMTcuMTMyNkwxNi4wODg1IDEzLjg1OTJMNyAxMy44NTkyWiIgZmlsbD0id2hpdGUiIC8+PC9zdmc+) ![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0iYXJyb3ctaG92ZXIiIHZpZXdib3g9IjAgMCAyNiAyNiIgZmlsbD0ibm9uZSI+PGNpcmNsZSBjeD0iMTMiIGN5PSIxMyIgcj0iMTIiIGZpbGw9IiNGRkVCQUEiPjwvY2lyY2xlPjxwYXRoIGQ9Ik03IDEzLjg1OTJMNyAxMi4yMDYyTDE2LjA4ODUgMTIuMjA2MkwxMy42Mjk1IDguODY3NDNMMTUuMzYwNyA4TDE5IDEzLjAwODJMMTUuMzgwMyAxOEwxMy42Mjk1IDE3LjEzMjZMMTYuMDg4NSAxMy44NTkyTDcgMTMuODU5MloiIGZpbGw9ImJsYWNrIiAvPjwvc3ZnPg==)

## Interpretability for all types of AI models

[Talk to our team](/request-demo)

[![](/_astro/modality-llms.BLVC1VFU.webp)LLMs ![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjggMjYiIGZpbGw9Im5vbmUiIGFyaWEtaGlkZGVuPSJ0cnVlIiBkYXRhLWFzdHJvLWNpZC0yem5lYnRhNz48cGF0aCBkPSJNMTIuNSAxLjM5MzU1TDI1IDEyLjY0MzVMMTIuNSAyMy44OTM1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjMuNzUiIGRhdGEtYXN0cm8tY2lkLTJ6bmVidGE3IC8+PHBhdGggZD0iTTIzLjc1IDEyLjY0MzFMMCAxMi42NDMxIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjMuNzUiIGRhdGEtYXN0cm8tY2lkLTJ6bmVidGE3IC8+PC9zdmc+)](/language) [![](/_astro/modality-life-sciences.DwG4ynuo.webp)Life Sciences ![](data:image/svg+xml;base64,PHN2ZyB2aWV3Ym94PSIwIDAgMjggMjYiIGZpbGw9Im5vbmUiIGFyaWEtaGlkZGVuPSJ0cnVlIiBkYXRhLWFzdHJvLWNpZC0yem5lYnRhNz48cGF0aCBkPSJNMTIuNSAxLjM5MzU1TDI1IDEyLjY0MzVMMTIuNSAyMy44OTM1IiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjMuNzUiIGRhdGEtYXN0cm8tY2lkLTJ6bmVidGE3IC8+PHBhdGggZD0iTTIzLjc1IDEyLjY0MzFMMCAxMi42NDMxIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjMuNzUiIGRhdGEtYXN0cm8tY2lkLTJ6bmVidGE3IC8+PC9zdmc+)](/life-sciences)
