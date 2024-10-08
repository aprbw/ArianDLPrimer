Just a hidden of list things that I think has potential to be explored about.

# Extending Convolution

Basics on convlution:

* AlexNet, Convolution, CNN: https://www.youtube.com/watch?v=UZDiGooFs54
* https://github.com/vdumoulin/conv_arithmetic
* https://en.wikipedia.org/wiki/Convolution_theorem

Note that the similarity of convolution with cellular automata (CA). So one way to explore is to see what other exciting stuff that people have done with CA:

* Basics on CA: https://en.wikipedia.org/wiki/Cellular_automaton and the underlying theory of https://en.wikipedia.org/wiki/Automata_theory . CA is big deal because it relates to the underlying theory of computation in general https://en.wikipedia.org/wiki/Abstract_machine and even more broadly: https://en.wikipedia.org/wiki/Artificial_life which goes into full circle back to AI.
* MNCA: multiple neighborhoods cellular automata. The authoritative figure for this is Slackermanz: https://slackermanz.com/ . Check the YouTube https://www.youtube.com/watch?v=xyx5C40HpQM 
* Lenia: CA but continous instead of discrete https://arxiv.org/abs/2005.03742 https://www.youtube.com/watch?v=7-97RhAZhXI
* Particle Life: [reformulation of CA](https://youtu.be/Z_zmZ23grXE?si=DUSOc9f8xnDHB6Rt) using [n-body](https://en.wikipedia.org/wiki/N-body_problem#Other_n-body_problems) https://www.ventrella.com/Clusters/ and a YouTube viz and explainer https://youtu.be/p4YirERTVF0?si=32_mmGty--kFmSw6 that includes demo and implementation https://particle-life.com/ 

# Rendering density / distribution / energy

Some claims that the heart of DL is simply a a probability density (or distribution, or energy function) estimation problem.

It seems that neural rendering and radiance field are really good at these tasks.

Why not use them to estimate the distribution directly?
