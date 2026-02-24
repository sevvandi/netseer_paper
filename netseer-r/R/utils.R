library(igraph)
library(feasts)

triangle_density <- function(gr) {
  sum(igraph::count_triangles(gr)) / (igraph::vcount(gr) * (igraph::vcount(gr) -
                                                              1) * (igraph::vcount(gr) - 2) / 6)
}

if ("feasts" %in% .packages()) {} 

#' Generates a bigger graph using either linear or exponential growth.
#'
#' Wrapper around 'generate_graph_linear' and 'generate_graph_exp'.
#'
#'@param num_graphs The number of graphs to be generated.
#'Default set to \code{15}.
#'@param gr The input graph to generate the next graph. If set to \code{NULL}
#'a graph using \code{igraph::sample_pa} is used as the input graph.
#'@param del_edge The proportion of edges deleted from the input graph. Default
#'set to \code{0.1}.
#'@param new_nodes The proportion of nodes added to the input graph. Default
#'set to \code{0.1}.
#'@param edge_increase The proportion of edges added to the input graph. Default
#'set to \code{0.1}.
#'@param mode The method the graphs grow in.
#'Either \code{"linear"} for linear creation, or \code{"exp"}.
#'Default set to \code{"exp"}.
#'
#'@return A graph list.
#'
#'@examples
#'set.seed(1)
#'gr <- generate_graph_list(12, 0.1, 0.1, 0.1, "exp")
#'gr
#'
#'@export
generate_graph_list <- function(num_graphs = 15,
                                del_edge = 0.1,
                                new_nodes = 0.1,
                                edge_increase = 0.1,
                                mode = "exp") {
  graphlist <- list()
  graphlist[[1]] <- gr <-  igraph::sample_pa(5, directed = FALSE)

  if (mode == "exp") {
    for (i in 2:num_graphs) {
      gr <-  generate_graph_exp(gr, del_edge , new_nodes, edge_increase)
      graphlist[[i]] <- gr
    }

  } else {
    for (i in 2:num_graphs) {
      gr <-  generate_graph_linear(gr, del_edge, new_nodes, edge_increase)
      graphlist[[i]] <- gr
    }

  }

  graphlist
}
