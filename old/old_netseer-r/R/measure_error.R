#' Gives an error measurement for predicted graphs
#'
#' This function compares the predicted graph with the actual and comptues the
#' vertex and edge error as a proportion
#'
#' @param actual The ground truth or actual graph.
#' @param predicted The predicted graph.
#'
#' @return The vertex error and edge error as a proportion.
#'
#' @examples
#' data(syngraphs)
#' # Taking the 20th graph as the actual and the 19th graph as predicted.
#' measure_error(syngraphs[[20]], syngraphs[[19]])
#'
#'
#' @export
measure_error <- function(actual, predicted) {
  vertex_err <- abs(igraph::vcount(actual) - igraph::vcount(predicted)) /
    igraph::vcount(actual)
  edge_err <- abs(igraph::ecount(actual) - igraph::ecount(predicted)) /
    igraph::ecount(actual)

  output  <- list(vertex_err = vertex_err, edge_err = edge_err)

}
