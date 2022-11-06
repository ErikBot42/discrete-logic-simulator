![.](https://github.com/ErikBot42/discrete-logic-simulator/actions/workflows/rust.yml/badge.svg)
# discrete-logic-simulator
Logic simulator for tick based games such as VCB

## Running
Requires the nightly rustc compiler.

| file | description |
| --- | --- |
| `src/logic.rs` | main simulation (the interesting stuff) |
| `src/blueprint.rs` | parse VCB blueprints into gate graphs |
| `src/raw_list.rs` | very unsafe but fast buffer |
| `benches/benchmark.rs` | benchmark sim and parser |
| `test_files/` | VCB blueprints to test with |

## Simplified algorithm for a subset of the implementations
It is an event based simulation.
Each gate and cluster/junction contains a counter, if an input activates, it increments, and if it deactivates it decrements.
Since gates only depend on the number of inputs this works.
It has a list of gates that need to update.
Each gate in that list is iterated through and new state is calculated, if it changes, this is propagated to the outputs, and the other gates get added to a new update lists.
To make sure each gate only gets added once, each gate has a flag on it that stores whether it is in the list.
Since every gate is only connected to clusters and every cluster is only connected to gates, gates can be updated in any order and it does not need to keep a double buffer of old/new states.
