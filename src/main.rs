#[allow(clippy::upper_case_acronyms)]


pub mod logic;
use crate::logic::*;

use std::time::Instant;
fn main() {
    let mut network = GateNetwork::default();
    let c1 = network.add_cluster();
    //let c2 = network.add_cluster();
    let g1 = network.add_gate(Gate::new(GateType::NOR, [c1].to_vec()), [c1/*,c2*/].to_vec());

    network.add_all_gates_to_update_list();
    println!("{:#?}",network);
    network.update();
    println!("{:#?}",network);
    
    network.update();
    println!("{:#?}",network);
    let iterations = 10_000_000;
    println!("running {} iterations",iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        network.update();
    }
    let elapsed_time = start.elapsed().as_millis();
    println!("{:#?}",network);
    println!("running {} iterations took {} ms",iterations, elapsed_time);
    println!("done");


    //let g1 = 
}
