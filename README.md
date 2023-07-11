The Decision Transformer is an architecture that models reinforcement learning as a sequential modeling problem and attempts to serve as a replacement for traditional offline reinforcement learning methods. Decision Transformers match the state-of-the-art performance on Atari and OpenAI Gym benchmarks. It also shows an ability to stitch together different segments or parts of training data to generate novel sequences that achieve the desired return. With this code, I investigate the Decision Transformer’s ability to stitch by training it on segmented graph data and seeing whether, at test time, it can stitch together seen segments to generate the shortest path in the graph.

The original decision transformer paper can be found below:

# Decision Transformer

Lili Chen\*, Kevin Lu\*, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas†, and Igor Mordatch†

\*equal contribution, †equal advising

A link to our paper can be found on [arXiv](https://arxiv.org/abs/2106.01345).
