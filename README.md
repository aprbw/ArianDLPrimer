# Arian's DL primer

This is a list of what to learn if you want to do deep learning based on my own personal opinion. My biases are:
* I like neuroscience
* I am not as good at math and programming as I would like to be (yet)
* I like [pretty pictures](https://betterimagesofai.org/)
* I did [academia](https://researchrepository.rmit.edu.au/esploro/outputs/9922229712001341)
* I started with [traffic forecasting](https://github.com/aprbw/traffic_prediction), so there are more stuff on timeseries and graph
* This is list was made / updated around 2nd half of 2024. Some things might get outdated in few years (or less).
* (I haven't actually read everything, this also works as my reading list.)
* I like a list that started with Khan Academy and ended with Category Theory

More about me: https://www.arianprabowo.com/

![Nine small images with schematic representations of differently shaped neural networks, a human hand making a different gesture is placed behind each network.](https://github.com/user-attachments/assets/97f9123e-f9cf-4a0e-84a8-61a3d4009fdd)

<span><a href="https://www.burg-halle.de/en/xlab">Alexa Steinbrück</a> / <a href="https://www.betterimagesofai.org">Better Images of AI</a> / Explainable AI / <a href="https://creativecommons.org/licenses/by/4.0/">Licenced by CC-BY 4.0</a></span>

## An appetizer board to whet your whistle
Don't expect any real learning to be happening.

Videos:
* CGP Grey How AI learn https://youtu.be/R9OHn5ZF4Uo?si=twh2c7noeizD21UB https://youtu.be/wvWpdrfoEv0?si=wvuCZ9Ol5O_3crpD
* 3b1b https://www.3blue1brown.com/topics/neural-networks
* AlexNet / Computer Vision (CV) / Convolution (conv) https://youtu.be/UZDiGooFs54?si=nW3b5gWOcmF2I4i4
* Reinforcement Learning (RL): https://youtu.be/Dw3BZ6O_8LY?si=SnDofUxoy4_kC7u9 https://youtu.be/JgvyzIkgxF0?si=gUXg6r8DsXTJbTBk https://youtu.be/MgowR4pq3e8?si=CzbkdpJ5IhzWarlj

Interactive stuff:
* Decision Tree http://www.r2d3.us/visual-intro-to-machine-learning-part-1/
* Unsupervised / clusterting algorithms: K-means https://www.naftaliharris.com/blog/visualizing-k-means-clustering/ DBScan https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/
* A multilayer perceptron ([MLP](https://en.wikipedia.org/wiki/My_Little_Pony:_Friendship_Is_Magic)) https://playground.tensorflow.org/
* Visualisation of loss landscape: https://losslandscape.com/
* Understanding LLM https://bbycroft.net/llm

Reading:
* Paper and gifs explaining convolution. https://github.com/vdumoulin/conv_arithmetic
* ML workflow https://medium.datadriveninvestor.com/my-machine-learning-workflow-7576f7dbcef3
* List of different topics https://paperswithcode.com/sota
* Feature visualization in Computer Vision (CNN) https://distill.pub/2017/feature-visualization/
* Just look at figure 2 https://arxiv.org/abs/1506.02078
* I think it is always a great idea to learn the "meta" level of that subject. In the case of deep learning, how it interacts with the rest of the world: which industry it is affecting, will affect, or could potential affect; economics, culture, society, politics, and ethics. Here is a good summary: https://www.stateof.ai/ .

## Prerequisite (from literal zero)
1. Learn the most advanced math that is typically available in hig hschool (Like IB Math HL) : https://www.khanacademy.org/
1. Learn Python and the typical libraries (NumPy, SciPy, pandas, scikit-learn, matplotlib [properly](https://matplotlib.org/matplotblog/posts/pyplot-vs-object-oriented-interface/), Vega-Altair)
1. A Neural Network in 11 lines of Python https://iamtrask.github.io/2015/07/12/basic-python-network/
1. The famous Andew Ng's Coursera https://www.coursera.org/specializations/machine-learning-introduction
1. Yann LeCun's NYU Deep Learning course https://atcold.github.io/NYU-DLSP21/
1. Bishop's PRML https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf
1. Learn how to read papers. Start by listening to people talking about a paper, and then re-read it after you watch the talk. https://www.youtube.com/c/yannickilcher

## Tools and practial tips
* [PyTorch](https://pytorch.org) path (there would be similar ones for [TensorFlow](https://www.tensorflow.org/) and [JAX](https://github.com/google/jax)
  * Do the basic tutorials as much / little as you want:
    * https://pytorch.org/tutorials/beginner/basics/intro.html
    * https://pytorch.org/tutorials/beginner/introyt.html
    * https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
  * Make sure you have a good idea on what is happening, at least, one layer under the hood of whatever abstraction level you are working on. For me who mostly work on architectures and framework, it would be this: https://pytorch.org/tutorials/beginner/nn_tutorial.html .
  * PyTorch Lightning https://lightning.ai/pytorch-lightning https://towardsdatascience.com/awesome-pytorch-lightning-template-485a75c2f47e
  * TensorBoard https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
* Experiment management https://wandb.ai/
* Coding deep learning is a bit different than normal programming. In academia, usually you just want your code to fully run once, to write a paper. So you might not over-invest in maintainability. On the other hand, things tend to fail silently, like the loss are just not going down. And there won't be any error / warning message that you will help you narrow things down. So a different approach is required. Here is a great [Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/).

## Time Series
* Amazon's Big TS tutorial https://www.amazon.science/videos-and-tutorials/forecasting-big-time-series-theory-and-practice
* Swiss AI Lab tutorial: https://gmlg.ch/tutorials/graph-based-processing/ecml-2023 https://arxiv.org/abs/2310.15978
* https://pytorch-geometric-temporal.readthedocs.io/
* Explainable AI for Time Series Classification: A Review, Taxonomy and Research Directions https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9895252

## Basic Concepts
* Universal Approximation Theorem https://medium.datadriveninvestor.com/how-ai-works-explained-with-an-analogy-from-finance-9d89a919cd74
* [Manifold Hypothesis](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) and NeuralODE
* Yann LeCun
	* Self-Supervised Learning https://arxiv.org/pdf/2304.12210
	* Energy Based Models https://yann.lecun.com/exdb/publis/orig/lecun-06.pdf (also taught in his course https://atcold.github.io/NYU-DLSP21/)
	* JEPA https://openreview.net/pdf?id=BZ5a1r-kVsf
* Geometric Deep Learning https://geometricdeeplearning.com/
* Ethics https://medium.com/swlh/dataset-can-only-be-unbalanced-not-racist-but-humans-can-5c522590efce

## Useful Concepts
* TRLML: Technology readiness levels for machine learning systems https://www.nature.com/articles/s41467-022-33128-9
* Nervous System (biology): 
	* Action potential, synapse, connectome
	* synaptic plasticity, Hebbian theory, why "a brain that heals easily cannot learn".
	* https://en.wikipedia.org/wiki/Functional_specialization_(brain) and https://en.wikipedia.org/wiki/Place_cell
* **GOFAI** Good old fashioned artificial intelligence and Symbollic AI
* **Non-Backprop ANN**: Kohonen's self-organizing map, Boltzmann Machine, Helmholtz Machine, Spiking Neural Network
* **Exotic architecture** : Bayesian Neural Nets, KAN: Kolmogorov-Arnold Networks https://arxiv.org/pdf/2404.19756, Neural Rendering: https://www.science.org/doi/10.1126/science.aar6170 https://www.matthewtancik.com/nerf
* Karl Friston Free energy principle / predictive coding / active inference / Anil Seth: The hallucination of consciousness
* A statistical perspective of deep learning https://deeplearningtheory.com/
* Information Theory https://bayes.wustl.edu/etj/articles/theory.1.pdf
* Type theory and functional programming https://colah.github.io/posts/2015-09-NN-Types-FP/
* Category Theory https://www.youtube.com/playlist?list=PLSdFiFTAI4sQ0Rg4BIZcNnU-45I9DI-VB

## Other lists
* MIT 6.S191 Introduction to Deep Learning http://introtodeeplearning.com/
* Just know stuff. (Or, how to achieve success in a machine learning PhD.) from an Oxford PhD graduate on Neural Differential Equations https://kidger.site/thoughts/just-know-stuff/
* https://algorithmiclens.substack.com/p/a-deep-dive-into-modern-deep-learning
* EleutherAI Deep learning for dummies cookbook https://github.com/EleutherAI/cookbook
* https://deep-learning-drizzle.github.io/ from https://www.reddit.com/r/learnmachinelearning/wiki/resource/
* https://www.pythondiscord.com/resources/?topics=data-science
* https://distill.pub/
* http://visxai.io/
* EleutherAI Math4ML Flowchart https://eleutherai.notion.site/Notion-Flowchart-Generator-48cb82f37e8c43d99331d29586f31dfc#45785c6e9b394d63a02f7b834e82f17f

