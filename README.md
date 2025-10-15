# network_inference_attention

## The Connectome Pays Attention

## Abstract
We propose a functional indeterpreation of a 'connectome', broadly defined, as the self-attention mask over neural tokens. 
A connectome is a comprehensive map of neural connections in the brain, and may be thought of as its "wiring diagram". 
These maps are available in varying types displaying different levels of detail such as functional, trasncriptomic. 
We show that connectomes can be seen as the result of optimizing various self-supervised objective functions using a diagonal masking self-attention. 
The different types of connectomes emerge from combinations of (1) the types of embeddings of neural tokens provided as input to the models and (2) the particular
flavor of self-supervised objectives. For example, we reproduce so-called functional connectomes by 
Wired connecotmes aris from using transcriptomic embeddings of each neuron and training to reconstruct them.

In causal language modeling, tokens represent elements of a sequence and the task is next-token prediction and the  so a casual mask is applied to the attention matrix to

The most general representation of a connectome is a weighted directed graph where the nodes are neurons and the edges are some quantifiable metric of the connectednes between any two neurons (synapse, correlation, etc). 
This can be displayed as an adjacency matrix of shape NxN which we suggest may be interpreted as an attention matrix over neural tokens. 

**Example project:** Can we recover the underlying connectome from functional data (i.e. neural activity)  in the nematoda _C. elegans_? 
