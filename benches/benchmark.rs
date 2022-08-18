use criterion::{black_box, criterion_group, criterion_main, Criterion};


use logic_simulator::fibonacci;

use logic_simulator::blueprint::*;

fn criterion_benchmark(c: &mut Criterion) {
    let mut parser_intro = BlueprintParser::default(); 
    let mut board_intro = parser_intro.parse("KLUv/WBAH00SAAQDAAAAAGZ4jv+hmFYqNUEAAAAATTg+Lkdd/5L/Y/8A/8ZjOE1HY/+frnT/MNn/Yor/gW6osQNRJNEFzEBDVTYAEgADAIAHCEoAIAACAVBAKAVH0EQDFcjIMAItIWOFtdMsyib1U+G1J34B/jfRybWBl8n3aXwSyVe19v5nnwm6m7qB2cp+9Y2978H5OWs9Fk/io/pket64EP/76piaNTcu/X9ZduD7AU3Zs/u+T8CdmWNOfUPMoxv7Dz/aMCpDM+pxBGTaX5nm2boUva+lgnzCKI4iWTUm1a0N+t4CGby6+7wK8hnhNNfi301xH8Poja7CcIB4cwN9ngpcxi03wa2+1AtehkWE0/jNfmLfdROkobGe1fdzM4ZVn2VM1g7KT8RstyOveC+7jEt5y+i6WYFb78eS4VM/UM70Xhyr2TxtXtiWD/o8KjbM7z0FzwtkdpEAh/JihGcH3kxNM5yI6dXi/sXsBtFkypPXByvYilBI+yVQf6/tm1VTXmBvVJ1t8MfgvTqT9EGZrSVOf1eq29prinXqlwjsaFK78+BQam2aVWNV+JavyM3a10qtgQjVbIOn83KZNl4ScqkS51XZukj6d2Acbbsln5e41T9CZBbloqhNiBf4l6dL6+UgDKHpZ91LI6CMeWisakh6lUfCknmpHs/OYtbKraG21LpuKPtqMJAWt+atvk0Dr7b3yUP5fi2hRLwJe4lr3/95DFXOolqUahO+deilG0wXmJig3PUpdgW0eFVnF69aPBCPfVbuafX6qXbC4/lvDwIAAAAwAAAAAgAAACsAAAACAAAAQCAAAAIAAAAAAAEA");

    let mut parser_bcd_count= BlueprintParser::default(); 
    let mut board_bcd_count= parser_bcd_count.parse("KLUv/WAIG40WANQCAABj8v//KjVB/y5HXf9NOD7/ZniO/6IAMNn/xmP/Yor/pgD//650YoouR13/gayo0QeQACEFRhkkpbwKEsAjAAEvIAJPMIUmBIBgwgkAAiQB1gE3nHx77w1qUwLONuKnqpvla2Wbmf/JVx0tY+ZTc0KFCyYLvvYPXyOmqLorbJuszy/sCb+8kJ808mKnXzKmv+TmZw9b3ivHGWzhgti0/rCj9lB2L5xsM+RFwBeIs/R9y1zsD2gZi87hHv/Cdu4S/B/cMsJrlFtoBp2N5Gc7HA2H5Gv5TjN+vVDn8p+wQLDwIZ5xkgt+nUUjOz9TaGcfoBk/1Z5fvCHsyT394DZXdlRi0YuX61iPxqf+vB21v8vLXMdOfuzeOHJkjpXDwmfR1rfUfOHdq03sZ+EL1/cQ6S+PO/iU//btQvhGf+atkt97SwTuDx/3APBN1ONzv8j0AD7xbt72NXTk951f0lmxROkGdtduG+I61VG0+aU/MJy2eNnnI3SNcVF8bcBU346Dl5v5oQNnE/2UjmKneZKfn1+qC/3c7uR3cCz2HjYws/pe2xu/707nop/8xnlEam7/T208VqT5R8fBzL0b6RFk7Qhq20Be9IzPKbe7M/cam5M78MhOHz92oEeHej7wVtq95GN/1ecQe8HCez5LsfYM+9VSvy8Gvp1rrWibfPGKL9ubnnePB9/SLwop35Q++mx8Ym/pQe1x97KPz8Mpb/Fq2KHl/KhhsIbgxtQW7cj61kV3l2LQru/Ubxitt9CtkrfI4j38Am2gByB9CAGa1f4LfCDgHHB+h5Yz8u5P2XVQfmXzOJrPQT12UV7c5bl6cPkhBtv5dXidrXewiZ4m1s/0vD0XG7zjuPD3NzYRjhO9Ji8/Rw/dszvZ7a3zc/Fh3ewNUH2nCaUXFcFODDuhabnRDilDVujXo7+x11daa23lQ36VkfBnYwGrwMTSsvs98AcCAAAAJwAAAAIAAAAuAAAAAgAAAAgcAAACAAAAAAABAA==");

    let mut c = c.benchmark_group("all");
    c.measurement_time(std::time::Duration::new(100,0));

    c.bench_function(
        "logic iterations intro", 
        |b| b.iter(|| board_intro.update()));
    c.bench_function(
        "logic iterations BCD count", 
        |b| b.iter(|| board_bcd_count.update()));
    c.bench_function(
        "circuit parser", 
        |b| b.iter(|| {
            let mut board = parser_intro.parse(black_box("KLUv/WBAH00SAAQDAAAAAGZ4jv+hmFYqNUEAAAAATTg+Lkdd/5L/Y/8A/8ZjOE1HY/+frnT/MNn/Yor/gW6osQNRJNEFzEBDVTYAEgADAIAHCEoAIAACAVBAKAVH0EQDFcjIMAItIWOFtdMsyib1U+G1J34B/jfRybWBl8n3aXwSyVe19v5nnwm6m7qB2cp+9Y2978H5OWs9Fk/io/pket64EP/76piaNTcu/X9ZduD7AU3Zs/u+T8CdmWNOfUPMoxv7Dz/aMCpDM+pxBGTaX5nm2boUva+lgnzCKI4iWTUm1a0N+t4CGby6+7wK8hnhNNfi301xH8Poja7CcIB4cwN9ngpcxi03wa2+1AtehkWE0/jNfmLfdROkobGe1fdzM4ZVn2VM1g7KT8RstyOveC+7jEt5y+i6WYFb78eS4VM/UM70Xhyr2TxtXtiWD/o8KjbM7z0FzwtkdpEAh/JihGcH3kxNM5yI6dXi/sXsBtFkypPXByvYilBI+yVQf6/tm1VTXmBvVJ1t8MfgvTqT9EGZrSVOf1eq29prinXqlwjsaFK78+BQam2aVWNV+JavyM3a10qtgQjVbIOn83KZNl4ScqkS51XZukj6d2Acbbsln5e41T9CZBbloqhNiBf4l6dL6+UgDKHpZ91LI6CMeWisakh6lUfCknmpHs/OYtbKraG21LpuKPtqMJAWt+atvk0Dr7b3yUP5fi2hRLwJe4lr3/95DFXOolqUahO+deilG0wXmJig3PUpdgW0eFVnF69aPBCPfVbuafX6qXbC4/lvDwIAAAAwAAAAAgAAACsAAAACAAAAQCAAAAIAAAAAAAEA"));
            board.update()
        }
        ));
    c.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
