---
title: 'Netseer: A Python and R package for Predicting Graph Structure via Adapted Flux Balance Analysis'
tags:
  - Python
  - R
authors:
  - name: Brodie Oldfield
    orcid: 0009-0000-4500-5006
    equal-contrib: true
    affiliation: 1
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: Data61, CSIRO, Australia
   index: 1
 - name: Independent Researcher, Country
   index: 2
date: 14 August 2025
bibliography: paper.bib
---

# Summary

**Netseer Intro
    - What goes into the summary:
    - Effectively it's the abstract,
    - This is pretty much rewording the abstract but shorter**

**Per JOSS, the summary is a high level functionality summary intended for a diverse, non specialist audience**

Growing networks are present in many fields, spanning from Telecommunications modeling to transport systems. `netseer` is a tool used to predict these modelled networks using an adapted form of Flux Balance Analysis (FBA). **--Look into FBA more/Maybe explanation. Keep it general**. FBA is adapted by ??modifying the optimisation problem to suit graph prediction??.

# Statement of need

**What goes into the statement of need:
    - Primarily, the purpose of the software.
    - Q. Why is Netseer needed?
    - A. Dynamic systems have imporant roles in life: E.g. Transport.
    - Q. Who is the intended audience?
    - A. Different fields that want to model graph growth.
    - A2. Telecommunications, Transport, Maybe Banking
    - Q. Are there products that would benefit from this?**

`netseer` is an R and Python-based package that uses an adapted form of Flux Balance Analysis to predict graph structures from time-series graphs. It is published on both CRAN[@netseerR] and PYPI [@netseerPy] under the `netseer` package.

As `netseer` is relevant to professions that occupy differing programming language ecosystems, `netseer` has both an R and Python implementation that act in parity. The underlying logic of netseer has a publication [@kandanaarachchi2025predictinggraphstructureadapted:2025].

**--Check if FBA use in Graph Prediction is novel** The use of FBA in graph prediction is a novel approach that, compared to existing methods, takes the weight vertices into consideration when performing graph predictions. This allows for more precise predictions compared to alternatives that don't take the possibility of weights changing between growth stages.

# Usage

`netseer` processes suitable discrete time-series graphs that are in an ordered sequence (e.g. by time) and grow over time **--Double check that growing is needed. I believe so.**.

**--Short description of how netseer compares to commonly used alternatives packages in this field.**

**--Are there any research projects using netseer.**

![A time-series graph growing, with a 1 step prediction by netseer.\label{fig:graph_grow}](assets/netseer.svg)

# Examples

Predicting a graph 1 step into the future.

``` Python
# Load the graphs. 
graph_list = utils.read_graph_list(path_to_graphs)

# Generate the predicted next graph
predicted_graph = predict.predict)graph(graph_list, h=1)
```

`path_to_graphs` is a list of paths to be loaded.
Increase `h` to predict more steps into the future.

# Acknowledgements

# References

**--Note: The cite for netseer is using the arxiv version.**
