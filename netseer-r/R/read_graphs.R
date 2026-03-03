#' Loads all graphs in a directory.
#'
#' This function loads all graphs from a directory to a list
#'
#' @param path_to_graphs The directory where graphs are contained.
#' @param format Formats supported by \code{igraph::read_graph}.
#'
#' @return A list of graphs in \code{igraph} format.
#'
#' @examples
#' \dontrun{
#' path_to_graphs <- normalizePath("./path/to/graphs/")
#' grlist <- read_graph_list(path_to_graphs, "gml")
#' grlist
#' }
#' @keywords internal
#' @export
load_graphs_dir <- function(path_to_graphs, format) {
  if (!(
    format %in% c(
      "edgelist",
      "pajek",
      "ncol",
      "lgl",
      "graphml",
      "dimacs",
      "graphdb",
      "gml",
      "dl"
    )
  )) {
    stop(
      "Invalid data format! Format needs to be one of the following: edgelist, pajek, ncol, lgl, graphml, dimacs, graphdb, gml, dl."
    )
  }
  # Add / to the path end if it doesn't.
  if (nchar(path_to_graphs) != "/") {
    path_to_graphs <- paste0(path_to_graphs, "/")
  }
  file_names <- list.files(path_to_graphs, full.names = TRUE, pattern = paste0('*.' , format))
  graphlist <- read_to_graph(file_names = file_names, format = format)
  graphlist
}

#' Loads graphs using a list of individual paths to each graph.
#'
#' This function adds graphs to a list by reading in a list of paths to individual graphs.
#'
#' @param graph_dir_list A list containing the absolute path to each individual graph.
#' @param format Formats supported by \code{igraph::read_graph}.
#'
#' @return A list of graphs in \code{igraph} format.
#'
#' @examples
#' \dontrun{
#' graph_dir_list <- list()
#' path1 <- normalizePath("graph1.gml")
#' path2 <- normalizePath("graph2.gml")
#' graph_dir <- append(graph_dir_list, path1 )
#' graph_dir <- append(graph_dir_list, path2 )
#'
#' grlist <- read_graph_list(graph_dir_list, "gml")
#' grlist
#' }
#' @keywords internal
#' @export
load_graphs_list <- function(graph_dir_list, format) {
  check_format(format)

  graphlist <- read_to_graph(file_names = graph_dir_list, format = format)
  graphlist
}


#' Loads graphs to memory using a desired method.
#'
#' This function loads graphs from the file system into the R environment.
#' There are two loading options:
#' Loading all files from a directory.
#' Loading individual graphs.
#'
#' @param use_directory The absolute path to a directory that contains graph files to load.
#' @param use_list A list of absolute paths to individual graph files to load.
#' @param format Formats supported by \code{igraph::read_graph}.
#'
#' @return A list of graphs in \code{igraph} format.
#'
#' @examples
#' \dontrun{
#' graph_dir_list <- list()
#' path1 <- normalizePath("graph1.gml")
#' path2 <- normalizePath("graph2.gml")
#' graph_dir <- append(graph_dir_list, path1 )
#' graph_dir <- append(graph_dir_list, path2 )
#'
#' grlist <- read_graph_list(graph_dir_list, "gml")
#' grlist
#' }
#' @export
load_graphs <- function(use_directory = NULL,
                        use_list = NULL,
                        format) {
  check_format(format)
  if (!is.null(use_directory) && !is.null(use_list)) {
    stop("Error: Both use_directory and use_list are used. Use only one.")
  }
  graphlist <- list()

  if (!is.null(use_directory)) {
    graphlist <- load_graphs_dir(use_directory, format)
  }
  if (!is.null(use_list)) {
    graphlist <- load_graphs_list(use_list, format)
  }
  if (is.null(use_directory) && is.null(use_list)) {
    stop("Error: Both use_directory and use_list are null. One needs to contain data.")
  }
  if (length(graphlist) == 0) {
    print("No graphs were loaded.")
  }

  graphlist
}


read_to_graph <- function(file_names, format) {
  graphlist <- list()
  for (i in 1:length(file_names)) {
    tempGraph <- igraph::read_graph(file_names[i], format = format)
    graphlist[[i]] <- tempGraph
  }
  graphlist
}

#' Saves either a single graph or list of graphs to disk.
#'
#' This function saves a single graph or list of graphs to a specified location on the file system in a specified format.
#'
#' @param graph Either a single igraph graph, or a list of igraph graphs.
#' @param file_path The Absolute path to save the graph/s to.
#' @param filetype The filetype extension to append to the graph file name, e.g. ".gml"
#' @param format Formats supported by \code{igraph::read_graph}.
#'
#' @return A list of graphs in \code{igraph} format.
#'
#' @examples
#' \dontrun{
#' library(igraph)
#' sample_graph  <- igraph::graph_from_literal(A-B, B-C)
#' path <- "/path/to/save/to/"
#' save_graphs(sample_graph, path, ".gml", "gml")
#' }
#' @export
save_graphs <- function(graph, file_path, filetype = ".gml", format) {
  # graph: What graph to save.
  # Save location:Filepath to save the data to
  # Format: Format that works with
  check_format(format)
  msg <- FALSE
  if (length(graph) == 0) {
    stop("Error: Supplied graph or graphlist has no length.")

  }
  if (class(graph) == "list") {
    filename <- paste0(file_path, filetype)
    write_graph(graph, filename, format)
    filename_check <- paste0(file_path, "1", filetype)
    if (file.exists(filename_check)) {
      msg <- TRUE
    }
  }
  if (class(graph) == "igraph") {
    for (i in 1:length(graph)) {
      filename <- paste0(file_path, i, filetype)
      write_graph(graph, filename, format)
    }
    filename_check <- paste0(file_path, "1", filetype)
    if (file.exists(filename_check)) {
      msg <- TRUE
    }
  }

  if (msg) {
    success_msg <- paste0("Saved to ", file_path)
    print(success_msg)
  }
  if (!msg) {
    error_msg <- paste0("Error: Failed to save file to ", file_path)
    print(error_msg)
  }

}

check_format <- function(format) {
  if (!(
    format %in% c(
      "edgelist",
      "pajek",
      "ncol",
      "lgl",
      "graphml",
      "dimacs",
      "graphdb",
      "gml",
      "dl"
    )
  )) {
    stop(
      "Error: Invalid data format! Format needs to be one of the following: edgelist, pajek, ncol, lgl, graphml, dimacs, graphdb, gml, dl."
    )
  }
}
