// Copyright (c) 2017 Will Kochanski
// @file   egraph/dp/interface.h
// @brief  Defines a general interface for dynamic programming problems.
// @author Will Kochanski
#pragma once

#include <tuple>                            // for pair
#include <vector>                           // for vector
#include <set>                              // for set
#include <unordered_set>                    // for unordered_set

#include "tbb/flow_graph.h"                 // for parallelism
#include "nonstd/variant.hpp"               // for variant

#include "egraph/dp/decomposition.h"        // for Decomposition, Cluster
#include "egraph/dp/boundary.h"             // for Boundary, MergingAlignment
#include "util/multi_queue.h"               // for MultiQueue


using std::pair;
using std::make_pair;
using std::vector;
using std::set;
using std::unordered_set;
using std::unordered_map;
using std::reference_wrapper;

using nonstd::variant;

namespace egraph { namespace dp {

// Defines a dynamic programming problem on a planar graph.
// Base is the type of the leaf case in the composition, e.g. "Edge".
// Config is the type used group partial solutions, e.g. a solution
//    restricted to the boundary of a subgraph
// Bucket is a type used to sort compatible Configs for merging, e.g.
//    a solution further restricted to the shared boundary between
//    two subgraphs.
// Solution is the solution type, e.g. EdgeSet.
template <class Base, class Config, class Bucket, class Solution>
class DynamicProgram {
 public:
  // A Config, annotated with cost and back-tracking information.
  struct Entry {
    size_t                                    cost;   // Cost of the config.
    Config                                    config; // The config.
    variant<Solution, pair<Entry&, Entry&>>   source; // The source of the config.

    Entry() {}
    // Construct an entry constructed from two source configs.
    Entry(size_t cost, Config config, pair<Entry&, Entry&> source) :
      cost(cost), config(config), source(source) {}
    // Construct an entry corresponding to a solution.
    Entry(size_t cost, Config config, Solution source) :
      cost(cost), config(config), source(source) {}
  };

 private:
  // Require an ordering to be defined on a pair of Entries.
  using BucketMap = unordered_map<Bucket, vector<reference_wrapper<Entry>>>;
  struct ComposeOrder {
    using EntryPair = pair<Entry&, Entry&>;
    bool operator() (const EntryPair&, const EntryPair&) const;
  };


  // The main computational step, compute all Entries for a given
  // Cluster, using a table of previous results.
  void handle_merge(Cluster<Base>&, vector<vector<Entry>>&);

  // Sort Entries into buckets based on the MergingAlignment
  void fill_buckets(vector<Entry>&, MergingAlignment, BucketMap&);

  // Compute a solution, backtracking from a particular Entry.
  Solution solution(Entry& start);

  Decomposition<Base> decomposition;

 public:
  // Construct a dynamic program structured around a decomposition.
  DynamicProgram(Decomposition<Base> &decomp) : decomposition(decomp) {}

  // Run the dynamic program, computing a solution.
  Solution run();
  // Run the dynamic program, using intel thread building blocks to allow
  // parallelism
  Solution run_parallel();

  // Return the entries for a cluster in the base case.
  virtual void base_case(Base, const Boundary&, vector<Entry>&) = 0;

  // Given a config, and a description of the merge, assign it a bucket.
  virtual Bucket to_bucket(const Config&, MergingAlignment) = 0;

  // Given a bucket on the left, list compatible buckets on the right.
  virtual vector<Bucket> match(const Bucket &left, MergingAlignment) = 0;

  // Compose two configs, producing all possible parent configurations.
  virtual vector<Config> compose(const Config &left, const Config &right, 
      MergingAlignment) = 0;

  // Compose two solutions, e.g. set union.
  virtual Solution sol_compose(const Solution &left, const Solution &right) = 0;
};

}}  // namespace egraph::dp

//=============================================================================
//= Implementation
//=============================================================================
namespace egraph { namespace dp {

template <class Base, class Config, class Bucket, class Solution>
bool DynamicProgram<Base, Config, Bucket, Solution>::ComposeOrder::operator()
  (const EntryPair& x, const EntryPair& y) const {
  size_t xcost = x.first.cost + x.second.cost;
  size_t ycost = y.first.cost + y.second.cost;
  return xcost > ycost;
}

template <class Base, class Config, class Bucket, class Solution>
void DynamicProgram<Base, Config, Bucket, Solution>::fill_buckets
  (vector<Entry>& entries, MergingAlignment alignment, BucketMap& bucketmap) {
  for (Entry& entry : entries) {
    Bucket bucket = to_bucket(entry.config, alignment);
    bucketmap[bucket].push_back(entry);
  }
}

template <class Base, class Config, class Bucket, class Solution>
Solution DynamicProgram<Base, Config, Bucket, Solution>::solution(Entry& start) {
  if (start.source.index() == 0) {        // this is a leaf node
    return nonstd::get<0>(start.source);
  } else {                                // this is an internal node
    Entry& left = nonstd::get<1>(start.source).first;
    Entry& right = nonstd::get<1>(start.source).second;
    return sol_compose(solution(left), solution(right));
  }
}


template <class Base, class Config, class Bucket, class Solution>
Solution DynamicProgram<Base, Config, Bucket, Solution>::run() {
  vector<vector<Entry>> entries(decomposition.size());

  // Handle leaf cases
  // lambda function to sort base case entries by cost
  auto cmp_entry = [](Entry a, Entry b) { return a.cost < b.cost; };
  for (Cluster<Base>& leaf : decomposition.leaves()) {
    base_case(leaf.value(), decomposition.get_boundary(leaf), entries[leaf.id]);
    std::sort(entries[leaf.id].begin(), entries[leaf.id].end(), cmp_entry);
  }

  // Handle internal cases
  int count = 0;
  for (Cluster<Base>& internal : decomposition.internals()) {
    count++;
    handle_merge(internal, entries);
  }

  assert(!entries.empty());           // there is a root node
  assert(!entries.back().empty());    // there is a solution at the root

  return solution(entries.back()[0]);
}

template <class Base, class Config, class Bucket, class Solution>
Solution DynamicProgram<Base, Config, Bucket, Solution>::run_parallel() {
  // Construct a flow graph representing computational dependency
  tbb::flow::graph flowgraph;
  tbb::flow::broadcast_node<tbb::flow::continue_msg> start(flowgraph);
  vector<tbb::flow::continue_node<tbb::flow::continue_msg>> continue_nodes;
  continue_nodes.reserve(decomposition.size());

  // The main storage table, each parallel node is passed
  // a reference to the entire table, but modifies only its own index
  // and only refers to indices that have been completed.
  vector<vector<Entry>> entries(decomposition.size());

  // Populate flow graph to handle leaf nodes
  for (const Cluster<Base>& leaf : decomposition.leaves()) {
    continue_nodes.emplace_back(flowgraph, [&] (const tbb::flow::continue_msg&) {
      base_case(leaf.base, entries[leaf.id]);
    });
    tbb::flow::make_edge(start, continue_nodes.back());
  }

  // Populate flow graph to handle internal nodes
  for (const Cluster<Base>& internal : decomposition.internals()) {
    continue_nodes.emplace_back(flowgraph, [&] (const tbb::flow::continue_msg&) {
      handle_merge(internal, entries);
    });
    auto& leftcontinue = continue_nodes[decomposition.left_child(internal)];
    auto& rightcontinue = continue_nodes[decomposition.right_child(internal)];
    tbb::flow::make_edge(leftcontinue, continue_nodes.back());
    tbb::flow::make_edge(rightcontinue, continue_nodes.back());
  }

  // Compute the results in parallel
  start.try_put(tbb::flow::continue_msg());
  flowgraph.wait_for_all();

  assert(!entries.empty());         // there is a root node
  assert(!entries.back().empty());  // there is a solution at the root
  return solution(entries.back()[0]);
}

template <class Base, class Config, class Bucket, class Solution>
void DynamicProgram<Base, Config, Bucket, Solution>::
  handle_merge(Cluster<Base> &parent, vector<vector<Entry>> &entries) {
  MergingAlignment alignment = decomposition.get_alignment(parent);

  Cluster<Base> left  = decomposition.left_child(parent);
  Cluster<Base> right = decomposition.right_child(parent);

  // Partition configs into buckets.
  BucketMap leftbuckets, rightbuckets;
  fill_buckets(entries[left.id], alignment, leftbuckets);
  fill_buckets(entries[right.id], alignment, rightbuckets);

  // Add valid pairs of buckets to the queue.
  // MultiQueue was created for this purpose, efficiently and lazily representing
  // the sorted union of set products of each pair of compatible buckets.
  // i.e. the underlying representation has type 
  //    SortedList<Pair<SortedList<A>, SortedList<A>>
  // but the interface behaves like
  //    SortedList<Pair<A, A>>
  util::MultiQueue<typename BucketMap::mapped_type, ComposeOrder> queue;
  for (auto& bucketpair : leftbuckets) {
    Bucket &leftbucket = std::get<0>(bucketpair);
    vector<reference_wrapper<Entry>> &leftentries = std::get<1>(bucketpair);
    if (leftentries.empty()) continue;
    for (Bucket &rightbucket : match(leftbucket, alignment)) {
      vector<reference_wrapper<Entry>>& rightentries = rightbuckets[rightbucket];
      if (rightentries.empty()) {
        continue;
      }
      queue.add_source(leftentries, rightentries);
    }
  }

  // Read unique Configs from the queue.
  // When the same Config appears twice in this process, it represents
  // two different solutions which are indistinguishable at this level.
  // We want to pass on only the solution with lower cost, and use the
  // Entry datatype to record the backtrack information.
  unordered_set<Config> seen_configs;
  while (!queue.empty()) {
    auto topentries = queue.top(); queue.pop();
    Entry& leftentry = topentries.first.get();
    Entry& rightentry = topentries.second.get();

    // Composition produces a vector of resulting Configs
    for (Config parentconfig : compose(leftentry.config, rightentry.config, 
          alignment)) {
      if (seen_configs.find(parentconfig) != seen_configs.end()) continue;

      // Prepare Entry annotations
      size_t cost = leftentry.cost + rightentry.cost;
      pair<Entry&, Entry&> source(leftentry, rightentry);
      entries[parent.id].emplace_back(cost, parentconfig, source);
      seen_configs.insert(parentconfig);
    }
  }
}

}}  // namespace egraph::dp
