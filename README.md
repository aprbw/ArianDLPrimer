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
* AlexNet / Computer Vision (CV) / Convolution (conv) by Welch Labs https://youtu.be/UZDiGooFs54?si=nW3b5gWOcmF2I4i4
* Reinforcement Learning (RL):
  * Yosh Trackmania https://youtu.be/Dw3BZ6O_8LY?si=SnDofUxoy4_kC7u9
  * Intro by Arxiv Insights https://youtu.be/JgvyzIkgxF0?si=gUXg6r8DsXTJbTBk
  * AlphaGO by Arxiv Insights https://youtu.be/MgowR4pq3e8?si=CzbkdpJ5IhzWarlj

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

(An alternative list of pre-req: https://roadmap.sh/ai-data-scientist)

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

![https://github.com/cruiseresearchgroup/DIEF_BTS/blob/main/snippet_6TS_plot.png](https://github.com/user-attachments/assets/14ccc7f9-28f7-4c18-a199-879ec2ad0ca0)

* https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
* Amazon's Big TS tutorial https://www.amazon.science/videos-and-tutorials/forecasting-big-time-series-theory-and-practice
* Swiss AI Lab tutorial: https://gmlg.ch/tutorials/graph-based-processing/ecml-2023 https://arxiv.org/abs/2310.15978
* https://pytorch-geometric-temporal.readthedocs.io/
* Explainable AI for Time Series Classification: A Review, Taxonomy and Research Directions https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9895252
* Spatio-temporal time series:
	* [CSUR 2018] Spatio-Temporal Data Mining: A Survey of Problems and Methods. Atluri et. al. https://arxiv.org/abs/1711.04710
	* [TKDE 2020] Deep Learning for Spatio-Temporal Data Mining: A Survey. Wang et. al. https://ieeexplore.ieee.org/abstract/document/9204396

## Get Dirty

I think this the a good time to get your hand dirty for the first time.
Earlier than this is better, but not later.
* A simple way would be to join competitions: https://www.kaggle.com/ , https://www.aicrowd.com/, and self-promo here: https://www.aicrowd.com/challenges/brick-by-brick-2024
* Another way is to simply try a new combination of method and dataset that people have not tried before. For example, there are always new timeseries forecasting algorithms: https://paperswithcode.com/task/time-series-forecasting/latest . Yet, usually, these new algorithms have not been tried on many of the timeseries dataset out there: https://forecastingdata.org/ .
* More experiments and analysis. In an ideal world, every paper out there would be accompanied by a full barrage of experiments and analysis. But due to limited resource (and also efficiency), usually not the entire barrage of analysis have been done. This will give you an opportunity to pick up a paper that you like, and do extra analysis that the paper did not do. The list of analysis includes:
  * Robustness: Hyper-parameter sensitivity
  * Robustness: Initial weight sensitivity
  * Robustness: Random seed for training data loader sensitivity
  * Robustness: Noise in training data
  * Robustness: Noise in input data during inference
  * Scaling: Data
  * Scaling: Compute
  * Error Analysis: Generate a residual plot or a confusion matrix to find the source of systematic error
  * Error Visualisation: Find the sample data with the largest amount of error and visualise the error
  * Attention Mechanism Visualization
  * Saliency Maps
  * Linear Probes and [Tuned Lens](https://github.com/AlignmentResearch/tuned-lens)
* Module combinations: Example, if there is a paper that uses an LSTM module, try to replace it with GRU or CNN or transformer.

## Basic Concepts

By basic, I do not mean that this where people should start. That would be the pre-requisite.
I do not mean that this is going to be easy or simple either. In fact, in my opinion, these are quite difficult and complex (or maybe I'm just not that smart).
I think people should have [gotten dirty](#get-dirty) with a couple of hands on project before getting to this stage. I think, only by then people are going to be able the appreciate the basics, as in the different theories on what are the elementary building blocks, the fundamentals.
Note that, as this is only a baby science, nobody knows what works, and how it works yet. But below is a collection of some theories that are most probably wrong, but contains some useful heuristic to help us along the way. I think it is worth slowing down and understanding every ideas in this section.

* Universal Approximation Theorem https://medium.datadriveninvestor.com/how-ai-works-explained-with-an-analogy-from-finance-9d89a919cd74
* [Manifold Hypothesis](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) and how RNN and ResNet are NeuralODE.
* Yann LeCun
	* Self-Supervised Learning https://arxiv.org/pdf/2304.12210
	* Energy Based Models https://yann.lecun.com/exdb/publis/orig/lecun-06.pdf (also taught in his course https://atcold.github.io/NYU-DLSP21/)
	* JEPA https://openreview.net/pdf?id=BZ5a1r-kVsf
* Geometric Deep Learning https://geometricdeeplearning.com/
* Ethics https://medium.com/swlh/dataset-can-only-be-unbalanced-not-racist-but-humans-can-5c522590efce
* It is good to be critical about the trends and current trends and think thorough about the interplay between the nature of the data, task, and methods. A good example from timeseies is: [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/pdf/2205.13504)

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
* https://www.fast.ai/
* Just know stuff. (Or, how to achieve success in a machine learning PhD.) from an Oxford PhD graduate on Neural Differential Equations https://kidger.site/thoughts/just-know-stuff/
* https://algorithmiclens.substack.com/p/a-deep-dive-into-modern-deep-learning
* EleutherAI Deep learning for dummies cookbook https://github.com/EleutherAI/cookbook
* https://deep-learning-drizzle.github.io/ from https://www.reddit.com/r/learnmachinelearning/wiki/resource/
* https://www.pythondiscord.com/resources/?topics=data-science
* https://distill.pub/
* http://visxai.io/
* EleutherAI Math4ML Flowchart https://eleutherai.notion.site/Notion-Flowchart-Generator-48cb82f37e8c43d99331d29586f31dfc#45785c6e9b394d63a02f7b834e82f17f

