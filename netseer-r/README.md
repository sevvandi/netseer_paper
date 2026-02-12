
<!-- README.md is generated from README.Rmd. Please edit that file -->

# Netseer

<!-- badges: start -->

[![R-CMD-check](https://github.com/sevvandi/netseer/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/sevvandi/netseer/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->
*Netseer* is a software package for predicting new
graphs from a given time series of graphs.

The underlying prediction algorithm combines time series modelling with
an adapted form of Flux Balance Analysis, an approach widely used in
biochemistry for reconstructing metabolic networks from partial
information. A comprehensive description of the algorithm is given in:

- Predicting Graph Structure via Adapted Flux Balance Analysis.  
  Lecture Notes in Computer Science (LNCS), Vol. 16370, 2026.  
  DOI:
  [10.1007/978-981-95-4969-6_27](https://doi.org/10.1007/978-981-95-4969-6_27);
  arXiv: [2507.05806](https://arxiv.org/abs/2507.05806)

## Installation

The `netseer` package is available for installation via CRAN:

``` r
install.packages("feasts")
install_packages("netseer")
```

Alternatively, `netseer` can be built from source from GitHub:

``` r
install.packages("feasts")
install.packages("remotes")
library("remotes")
remotes::install_github("sevvandi/netseer_paper/netseer-r")
```

When building from GitHub, a C++ compiler may be needed. For Windows and Mac: Install [RTools](https://cran.r-project.org/bin/windows/Rtools/rtools44/rtools.html). For Linux: Ubuntu `sudo apt install build-essential`, Fedora `sudo dnf group install "Development Tools"` etc.  
For Linux ensure `curl` is installed. For Ubuntu: `sudo apt install curl`, Fedora: `sudo dnf install curl-devel`  
TODO: mention R-tools is required under Windows; under Linux the curl-devel / curl-dev package system package should be first installed
TODO: I'll check the ubuntu install, May not need a dev version of curl, just the base package. As the dev curl would be under something like `libcurl4-openssl-dev`.

## Available Functions

| Function | Summary |
|----|----|
| `read_graph_list()` | Load user provided graphs. TODO: mention alpha-numeric order; also state in documentation |
| `predict_graph()` | Predict the next graph in a sequence. |
| `measure_error()` | Return the vertex error and edge error between two graphs. |
| `generate_graph_linear()` | Generate a time series of random graphs that grow linearly. |
| `generate_graph_exp()` | Generate a time series of random graphs that grow exponentially. |

Documentation for the above functions is available in the [Documentation
PDF](./docs/netseer.pdf)

## Example

Goal:

- Load 20 graphs from the file system.
- Use graphs 1 to 19 to predict the 20th graph.
- Compare the actual 20th graph to the newly predicted 20th graph.

Before starting, download the [example_graphs.zip](./example_graphs.zip) directory under
`/netseer-paper/netseer-r/`.  This directory contains 20 example graphs.
Extract the zip to your project root.

``` r
library("netseer")

# Load 20 graphs from the example_graphs directory.
## Replace ./example_graphs/ in file.path with a path to the example_graphs directory.
path_to_graphs <- file.path("./example_graphs/")
graph_list <- netseer::read_graph_list(path_to_graphs = path_to_graphs, format = "gml")

# Predict the 20th graph using graphs 1 to 19.  
## h=1 means predict 1 step into the future.
## weights_opt=8 selects method 8 for weight optimisation/selection?; see documentation for details
## TODO: clarify what weights_opt=8 means
predicted_graph <- netseer::predict_graph(graph_list[1:19], weights_opt=8, h=1)

# TODO: netseer::predict_graph() is quite noisy
# TODO: to avoid confusion, need to print out whether the modelling succeeded or failed

# Compare the 20th actual graph and the predicted 20th graph by checking the vertex and edge error.
vertex_err, edge_err <- netseer::measure_error(graph_list[[20]], predicted_graph[[1]])
print(vertex_err)
print(edge_err)

# TODO: "vertex_err, edge_err <- netseer::measure_error() currently doesn't work
```
