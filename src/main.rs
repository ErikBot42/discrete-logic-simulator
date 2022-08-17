
extern crate base64;
extern crate zstd;
extern crate colored;

#[allow(clippy::upper_case_acronyms)]

pub mod logic;
pub mod blueprint;
//use crate::logic::*;
use crate::blueprint::*;

//use std::time::Instant;
fn main() {
    let mut parser = BlueprintParser::default(); 
    //parser.parse("KLUv/SAwtQAAgE04Pv8uR13/AAAAAP9iiv8BAMbdEgIAAAAEAAAAAgAAAAMAAAACAAAAMAAAAAIAAAAAAAEA");
    parser.parse("KLUv/WBAH00SAAQDAAAAAGZ4jv+hmFYqNUEAAAAATTg+Lkdd/5L/Y/8A/8ZjOE1HY/+frnT/MNn/Yor/gW6osQNRJNEFzEBDVTYAEgADAIAHCEoAIAACAVBAKAVH0EQDFcjIMAItIWOFtdMsyib1U+G1J34B/jfRybWBl8n3aXwSyVe19v5nnwm6m7qB2cp+9Y2978H5OWs9Fk/io/pket64EP/76piaNTcu/X9ZduD7AU3Zs/u+T8CdmWNOfUPMoxv7Dz/aMCpDM+pxBGTaX5nm2boUva+lgnzCKI4iWTUm1a0N+t4CGby6+7wK8hnhNNfi301xH8Poja7CcIB4cwN9ngpcxi03wa2+1AtehkWE0/jNfmLfdROkobGe1fdzM4ZVn2VM1g7KT8RstyOveC+7jEt5y+i6WYFb78eS4VM/UM70Xhyr2TxtXtiWD/o8KjbM7z0FzwtkdpEAh/JihGcH3kxNM5yI6dXi/sXsBtFkypPXByvYilBI+yVQf6/tm1VTXmBvVJ1t8MfgvTqT9EGZrSVOf1eq29prinXqlwjsaFK78+BQam2aVWNV+JavyM3a10qtgQjVbIOn83KZNl4ScqkS51XZukj6d2Acbbsln5e41T9CZBbloqhNiBf4l6dL6+UgDKHpZ91LI6CMeWisakh6lUfCknmpHs/OYtbKraG21LpuKPtqMJAWt+atvk0Dr7b3yUP5fi2hRLwJe4lr3/95DFXOolqUahO+deilG0wXmJig3PUpdgW0eFVnF69aPBCPfVbuafX6qXbC4/lvDwIAAAAwAAAAAgAAACsAAAACAAAAQCAAAAIAAAAAAAEA");




    //parser.parse("KLUv/WBAH1UQAKQCAAAAAGZ4jv+hmFYqNUEAAAAATTg+Lkdd/5L/Y/8A/8Zj//9iiv+udP//gT2owRNQJKEFDkkzVjYBEoACAIAoMAkAAhoQACWogiMAAiU0ghEAjTx5LWGxwbUzC41hlaq5dtJL419SF9eWr71Lmadbvtra8c/WE8hr5cB2ZXh1x457cB9OjkU7n3RL5CpFY/OOGidYc8NAVksHJ7Kt55kFmm1NdauspC0X2sbzeCx4Divwt6JLDAYvXX0dPgrJpQ7syE0KcZsX0tLw1iIPpWdCH5csb9rTuwbuztz1T8Ft9EaXuXY9jWr1E6uuTxfNtkzvW507ZWgq38K8fGsM1CpBjUGtqBM0XmbHJG7whlfUd7a6akVud+mPrDfDtUbwgtl0wfiiUoNSWdrE/oMW34hd0JwSNWfkbc8SfVxgbESlbbBSy96d3eSg/hY7X39X5FPTabG975UCqJrwtjwolBub0GosHb71K9Sz8FrbNVkiMVr/bC4vojSWpFlKh7zKt46Rkq2Nc75bLB3NbqUe4ZiFYKHXJuSrqMn/WnWBRZfA/bDRomaPxQoeixgSX/DD7lYu0KP0WWtt4BrE1t2Xh5By/Z7UILuATdjU+XI78HEofuaCZJqbHC83Afo/7hCoWXOWrza5W/xecoPxQuHEGkBLsa6K7gU6cxC1csEX3iqJB1MvOWspTN5/ewIAAAAwAAAAAgAAACsAAAACAAAAQCAAAAIAAAAAAAEA");
    //parser.parse("KLUv/SAYwQAA/8Zj/6GYVv9NOD7/AAAAAC5HXf//Yor/AgAAAAIAAAACAAAAAwAAAAIAAAAYAAAAAgAAAAAAAQA=");
    //parser.parse("KLUv/SAIQQAALkdd//9iiv8CAAAAAQAAAAIAAAACAAAAAgAAAAgAAAACAAAAAAABAA==");
    //parser.parse("KLUv/WBEGC0DALAAAP/GY/+hmFb/TTg+/wAuR13//2KKQajgBQ0d0ItcABJ4C8FfLfh3CGqn6g2h28BiXkXD1eWqG64uV7thtXzVDavL1W64Wq52w1U1K2DdUGpD2nGpTFl/9BdgeEQeR+1fWuAHAgAAACEAAAACAAAAMQAAAAIAAABEGQAAAgAAAAAAAQA=");
    //parser.parse("KLUv/WBAH2UQAOQCAAAAAGZ4jv+hmFYqNUEAAAAATTg+Lkdd/5L/Y/8A/8Zj//9iiv9j/5///wBB/4FAqMETWCSxBQxJM1Y2ARKgAgBAKFAJAAIaEAAlqIIjAAIlNIIRAI00dUVi+VPxrGmGNdWrFX7VvGcCZVRL+QV0afn5wtttaH35bDXJvo6xemW4/GNHOLgPJ3QUex1zy7Zqe3j4ZiK1uIJA1K1pvuKcg4Hctdl5gJU0yOnbyDxON8j4t+lWNYuO7UvxX6IPmLnk4A64CSEd5oUUN6y18IM5hvsjmOWe1oSmodozV/1T0I3e+DIpWU2jXP3EiusTjNKhmdIOzZQ2NIttaV6+NR21wrX+v9b+ia18W7U9PHxjKnpcUbG1l7e4FqrLNsOxzmMRXL2I6GKvQViZ2oTWA0rL/hc0xsRa02wJs9zgy86N1NoGqyx0G6d0PGi+ZW7n70q5wh36spteJsi1CU/vQfRsb5qpsch+ga9GzdrUhq4BEfCg/LOpd8FcvoxkKYXzKt86RhK3Ms6fbhEDjLyVf4RjFoKFUJtQXosgKxapF1gvo91nihaL/XgxkLGoIfEFHeZ+doF9lD9rqI1bg239/XiIbC5scPpooaeTTS0vvxMLH+W7XAnJyabxJThu/ycObZyVZqGtTcjW2ZdtMF8I9XDBaGCs0ZN/Qc4cs1oxJQxvtcQDU0nItdwl72k+AgAAADAAAAACAAAAKwAAAAIAAABAIAAAAgAAAAAAAQA=");
    //parser.parse("KLUv/SCQvQMAcggbI5ArAMhhSWZsnfijT+C5S5JrveAkYl1y7xYWkeObps5JUmkfd3flRD4YCEf30gSN3f407xF8mYdUvyL7T7t1vk0ZIQWeLKy7uwvQtIXqpePSuhUvg7fCJeXsNfuYxcxLZ6Pz6gmqZxvsaAAIAgBHEOMDYAYCAAAACQAAAAIAAAAEAAAAAgAAAJAAAAACAAAAAAABAA==");
    //let mut network = GateNetwork::default();
    //let c1 = network.add_cluster();
    ////let c2 = network.add_cluster();
    //let g1 = network.add_gate(Gate::new(GateType::NOR, [c1].to_vec()), [c1/*,c2*/].to_vec());

    //network.add_all_gates_to_update_list();
    //println!("{:#?}",network);
    //network.update();
    //println!("{:#?}",network);
    //
    //network.update();
    //println!("{:#?}",network);
    //let iterations = 10_000_000;
    //println!("running {} iterations",iterations);
    //let start = Instant::now();
    //for _ in 0..iterations {
    //    network.update();
    //}
    //let elapsed_time = start.elapsed().as_millis();
    //println!("{:#?}",network);
    //println!("running {} iterations took {} ms",iterations, elapsed_time);
    //println!("done");


    //let g1 = 
}
