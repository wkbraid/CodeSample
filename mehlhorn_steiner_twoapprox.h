// Copyright (c) 2017 Will Kochanski
// @file   mehlhorn_steiner_twoapprox.h
// @brief  Implementation of Mehlhorn's two-approximation for Steiner Tree.
// @author Will Kochanski
#pragma once

#include <vector>                   // for vector
#include <tuple>                    // for tuple
#include "egraph/graph.h"           // for Graph
#include "egraph/extra/sptree.h"    // for SPTree
#include "egraph/extra/minor.h"     // for contract
#include "alg/prim_mst.h"           // for prim_mst
#include "util/table.h"             // for Table

using namespace egraph;
using egraph::extra::SPTree;
using egraph::extra::contract;

namespace alg {

// Implementation of Mehlhorn's two-approximation for Steiner Tree.
template <GraphType R>
DartSet mehlhorn_steiner(Graph<R> &graph, 
    const Table<Dart, size_t> &length, const NodeSet &terminals) {
  // Compute Voronoi regions
  SPTree sptree(graph, length, terminals);

  // Construct a distance network
  Graph<R> network = contract(graph, sptree.edgeset());

  Table<Dart, size_t> network_length(network.num_darts());
  for (Dart d : network.darts()) {
    network_length[d]  = length[d];
    network_length[d] += sptree.dist(graph.head(network.embed(d)));
    network_length[d] += sptree.dist(graph.tail(network.embed(d)));
  }

  // Compute MST of distance network
  DartSet network_mst = prim_mst(network, network_length).dartset();
  DartSet result;
  for (Dart d : network_mst) {
    result.insert(network.embed(d));
    for (Dart path : sptree.rootward_shortest_path(graph.head(network.embed(d)), graph))
      result.insert(path);
    for (Dart path : sptree.leafward_shortest_path(graph.tail(network.embed(d)), graph))
      result.insert(path);
  }
  return result;
}

}   // namespace alg
