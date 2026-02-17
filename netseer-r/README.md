
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

**TODO:** add a rudimentary but explicit dependency on the 'feasts' package, for example by adding a function to netseer that prints the version of feasts (and possibly versions of other packages)

The `netseer` package is available for installation via CRAN:

``` r
install.packages(c('feasts', 'netseer'))
```

Alternatively, `netseer` can be built from source from GitHub:

``` r
install.packages(c('feasts', 'remotes'))
library("remotes")
remotes::install_github("sevvandi/netseer_paper/netseer-r")
```

When building from GitHub, a C++ compiler may be needed. For Windows and Mac: Install [RTools](https://cran.r-project.org/bin/windows/Rtools/rtools44/rtools.html). For Linux: Ubuntu `sudo apt install build-essential`, Fedora `sudo dnf group install "Development Tools"` etc.  
For Linux ensure `curl` is installed. For Ubuntu/Debian: `sudo apt install curl`, Fedora/RHEL/CentOS: `sudo dnf install curl-devel`  

**TODO:** check the actual package name in Ubuntu/Debian; I suspect it's something like `curl-dev`

**TODO:** I'll check the ubuntu install, May not need a dev version of curl, just the base package. As the dev curl would be under something like `libcurl4-openssl-dev`.

## Available Functions

**TODO:** for the description of `read_graph_list()` below, state that graphs are loaded in alpha-numeric order

**TODO:** also state in documentation that `read_graph_list()` loads graphs are loaded in alpha-numeric order

| Function | Summary |
|----|----|
| `read_graph_list()` | Load user provided graphs.|
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

Before starting, download the [example_graphs.zip](./example_graphs.zip) and extract the zip to your project root.
The zip contains 20 example graphs.

**TODO:** **Complete (Success Output)** ensure that modelling success is indicated by `netseer::predict_graph()`; right now it's noisy with confusing output; the current output suggests that the modelling failed  

**TODO:** **Complete** `vertex_err, edge_err <- netseer::measure_error()` currently doesn't work

**TODO:** **Complete** ensure that the errors are positive in python.

**TODO:**  netseer::predict_graph() still doesn't print out success;  instead we get weird warning and error messages

**TODO:** link to where more info about `weights_opt` can be found (ie. state that the options are explained in the documentation)

**TODO:** ensure that `weights_opt` are actually documented in the documentation

**TODO:** reject invalid weights_opt

**TODO:** ensure that the weights_opt ids are the same in the R and Python versions

**TODO:** if there is no corresponding weights_opt id in Python, disable the functionality in R

**TODO:** ensure that the errors are the same (or very close) for the same weights_opt in R and Python versions


``` r
library("netseer")

# Load 20 graphs from the example_graphs directory.
## Replace ./example_graphs/ in file.path with a path to the example_graphs directory.
path_to_graphs <- file.path("./example_graphs/")
graph_list <- netseer::read_graph_list(path_to_graphs = path_to_graphs, format = "gml")

# Predict the 20th graph using graphs 1 to 19.  
## h=1 means predict 1 step into the future.
## TODO: weights_opt=8 selects method 8 for weight optimisation/selection?; see documentation for details
## TODO: clarify what weights_opt=8 means
## TODO: ensure documentation has a brief explanation of what each weights_opt means
## TODO: (ie. what does 1 mean, what does 2 mean, ...)
predicted_graph <- netseer::predict_graph(graph_list[1:19], weights_opt=1, h=1)

# TODO: netseer::predict_graph() is quite noisy
# TODO: to avoid confusion, need to print out whether the modelling succeeded or failed

# Compare the 20th actual graph and the predicted 20th graph by checking the vertex and edge error.
output <- netseer::measure_error(graph_list[[20]], predicted_graph[[1]])
print(output)

```





