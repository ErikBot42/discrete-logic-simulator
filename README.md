# discrete-logic-simulator
Logic simulator for tick based games such as VCB

| file | description |
| --- | --- |
| `src/logic.rs` | main simulation (the interesting stuff) |
| `src/blueprint.rs` | parse VCB blueprints into gate graphs |
| `src/raw_list.rs` | very unsafe but fast buffer |
| `benches/benchmark.rs` | benchmark sim and parser |
| `test_files/` | VCB blueprints to test with |
