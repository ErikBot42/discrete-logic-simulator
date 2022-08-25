
extern crate base64;
extern crate zstd;
extern crate colored;

use std::process::Command;
use logic_simulator::blueprint::Parser;

fn main() {
        let string = include_str!("../test_files/big_decoder.blueprint");
        //let string = include_str!("../test_files/small_decoder.blueprint");
        //let string = include_str!("../test_files/circle.blueprint");
        //let string = include_str!("../test_files/gates.blueprint");
        //let string = include_str!("../test_files/invalid_base64.blueprint");
        //let string = include_str!("../test_files/invalid_zstd.blueprint");

        let mut board = Parser::parse(string);
        loop {
            //print!("\x1B[0;0H");
            //board.print();
            board.update();
            //let mut child = Command::new("sleep").arg("0.5").spawn().unwrap();
            //child.wait().unwrap();
        }
}
