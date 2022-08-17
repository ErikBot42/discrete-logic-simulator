use criterion::{black_box, criterion_group, criterion_main, Criterion};


use logic_simulator::fibonacci;

use logic_simulator::blueprint::*;

fn criterion_benchmark(c: &mut Criterion) {
    let mut parser = BlueprintParser::default(); 
    let mut board = parser.parse("KLUv/WBAH00SAAQDAAAAAGZ4jv+hmFYqNUEAAAAATTg+Lkdd/5L/Y/8A/8ZjOE1HY/+frnT/MNn/Yor/gW6osQNRJNEFzEBDVTYAEgADAIAHCEoAIAACAVBAKAVH0EQDFcjIMAItIWOFtdMsyib1U+G1J34B/jfRybWBl8n3aXwSyVe19v5nnwm6m7qB2cp+9Y2978H5OWs9Fk/io/pket64EP/76piaNTcu/X9ZduD7AU3Zs/u+T8CdmWNOfUPMoxv7Dz/aMCpDM+pxBGTaX5nm2boUva+lgnzCKI4iWTUm1a0N+t4CGby6+7wK8hnhNNfi301xH8Poja7CcIB4cwN9ngpcxi03wa2+1AtehkWE0/jNfmLfdROkobGe1fdzM4ZVn2VM1g7KT8RstyOveC+7jEt5y+i6WYFb78eS4VM/UM70Xhyr2TxtXtiWD/o8KjbM7z0FzwtkdpEAh/JihGcH3kxNM5yI6dXi/sXsBtFkypPXByvYilBI+yVQf6/tm1VTXmBvVJ1t8MfgvTqT9EGZrSVOf1eq29prinXqlwjsaFK78+BQam2aVWNV+JavyM3a10qtgQjVbIOn83KZNl4ScqkS51XZukj6d2Acbbsln5e41T9CZBbloqhNiBf4l6dL6+UgDKHpZ91LI6CMeWisakh6lUfCknmpHs/OYtbKraG21LpuKPtqMJAWt+atvk0Dr7b3yUP5fi2hRLwJe4lr3/95DFXOolqUahO+deilG0wXmJig3PUpdgW0eFVnF69aPBCPfVbuafX6qXbC4/lvDwIAAAAwAAAAAgAAACsAAAACAAAAQCAAAAIAAAAAAAEA");
    c.bench_function(
        "logic iterations", 
        |b| b.iter(|| board.update()));
    c.bench_function(
        "circuit parser", 
        |b| b.iter(|| {
            let mut board = parser.parse(black_box("KLUv/WBAH00SAAQDAAAAAGZ4jv+hmFYqNUEAAAAATTg+Lkdd/5L/Y/8A/8ZjOE1HY/+frnT/MNn/Yor/gW6osQNRJNEFzEBDVTYAEgADAIAHCEoAIAACAVBAKAVH0EQDFcjIMAItIWOFtdMsyib1U+G1J34B/jfRybWBl8n3aXwSyVe19v5nnwm6m7qB2cp+9Y2978H5OWs9Fk/io/pket64EP/76piaNTcu/X9ZduD7AU3Zs/u+T8CdmWNOfUPMoxv7Dz/aMCpDM+pxBGTaX5nm2boUva+lgnzCKI4iWTUm1a0N+t4CGby6+7wK8hnhNNfi301xH8Poja7CcIB4cwN9ngpcxi03wa2+1AtehkWE0/jNfmLfdROkobGe1fdzM4ZVn2VM1g7KT8RstyOveC+7jEt5y+i6WYFb78eS4VM/UM70Xhyr2TxtXtiWD/o8KjbM7z0FzwtkdpEAh/JihGcH3kxNM5yI6dXi/sXsBtFkypPXByvYilBI+yVQf6/tm1VTXmBvVJ1t8MfgvTqT9EGZrSVOf1eq29prinXqlwjsaFK78+BQam2aVWNV+JavyM3a10qtgQjVbIOn83KZNl4ScqkS51XZukj6d2Acbbsln5e41T9CZBbloqhNiBf4l6dL6+UgDKHpZ91LI6CMeWisakh6lUfCknmpHs/OYtbKraG21LpuKPtqMJAWt+atvk0Dr7b3yUP5fi2hRLwJe4lr3/95DFXOolqUahO+deilG0wXmJig3PUpdgW0eFVnF69aPBCPfVbuafX6qXbC4/lvDwIAAAAwAAAAAgAAACsAAAACAAAAQCAAAAIAAAAAAAEA"));
            board.update()
        }
        ));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
