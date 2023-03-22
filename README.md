![.](https://github.com/ErikBot42/discrete-logic-simulator/actions/workflows/rust.yml/badge.svg)
# discrete-logic-simulator
Logic simulator for tick based games such as VCB with a CLI based viewer.

![clip](https://user-images.githubusercontent.com/63870842/201472068-4dfb62d5-0c28-48c0-ae3d-54ddf9415810.gif)

(Video looks a bit weird because making gifs is hard)

```
Usage: logic_simulator [OPTIONS] <--blueprint-file <BLUEPRINT_FILE>|--blueprint-string <BLUEPRINT_STRING>|--world-file <WORLD_FILE>> <MODE> [IMPLEMENTATION]

Arguments:
  <MODE>
          What mode to run the program in

          Possible values:
          - emoji:     Print board using regular emojis
          - emoji-vcb: Print board using vcb emoji
          - print:     Print initial state of board
          - run:       Run and display state of board
          - bench:     Run a number of iterations and print time

  [IMPLEMENTATION]
          What implementation to use

          [default: reference]

          Possible values:
          - reference:   Used to compare performance and check correctness
          - scalar-simd: Pack values inside single register instead of simd
          - simd:        Update gates with simd

Options:
  -f, --blueprint-file <BLUEPRINT_FILE>      Filepath to VCB blueprint string
  -b, --blueprint-string <BLUEPRINT_STRING>  VCB blueprint string
  -w, --world-file <WORLD_FILE>              Filepath to VCB world
  -l, --legend                               Print legend with emoji
  -i, --iterations <ITERATIONS>              Iterations to run in bench [default: 10000000]
  -h, --help                                 Print help information (use `--help` for more detail)
```

## Simplified algorithm for a subset of the implementations
It is an event based simulation.
Each gate and cluster/junction contains a counter, if an input activates, it increments, and if it deactivates it decrements.
This works because the gate (AND, OR, XOR, NAND...) and cluster state only depend on the number of inputs.
It has a list of gates that need to update.
Each gate in that list is iterated through and new state is calculated, if it changes, this is propagated to the outputs, and the other gates get added to a new update list.
To make sure each gate only gets added once, each gate has a flag on it that stores whether it is in the list.
Since every gate is only connected to clusters and every cluster is only connected to gates, gates can be updated in any order and it does not need to keep a double buffer of old/new states.

## Installing the nightly rust compiler
Get rustup from your package manager (or https://rustup.rs/).
Install the nightly compiler
```
rustup default nightly
```
A number of nightly features are used, notably `portable_simd`
### Getting the code
Using git (or download zip of entire repo)
```
git clone https://github.com/ErikBot42/discrete-logic-simulator.git
```
### Running and compiling
```
cargo run
```
or
```
cargo build
./target/debug/logic_simulator
```
## Usage
Print help
```
cargo run -- --help
```
Example, run an included circuit
```
cargo run -- --blueprint-file test_files/intro.blueprint run
```
Running tests and criterion benchmarks
```
cargo test
cargo bench
```

### Files
| file | description |
| --- | --- |
| `src/logic.rs` | main simulation (the interesting stuff) |
| `src/logic/network.rs` | gate network management |
| `src/logic/gate_status.rs` | bit manipulation for the logic sim |
| `src/blueprint.rs` | parse VCB blueprints into gate graphs |
| `src/raw_list.rs` | very unsafe but fast buffer |
| `benches/benchmark.rs` | benchmark sim and parser |
| `test_files/` | VCB blueprints to test with |


