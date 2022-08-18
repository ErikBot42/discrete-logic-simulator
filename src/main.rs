
extern crate base64;
extern crate zstd;
extern crate colored;

use std::process::Command;
use logic_simulator::blueprint::*;

fn main() {
    //let mut board = BlueprintParser::default().parse("KLUv/WBAH00SAAQDAAAAAGZ4jv+hmFYqNUEAAAAATTg+Lkdd/5L/Y/8A/8ZjOE1HY/+frnT/MNn/Yor/gW6osQNRJNEFzEBDVTYAEgADAIAHCEoAIAACAVBAKAVH0EQDFcjIMAItIWOFtdMsyib1U+G1J34B/jfRybWBl8n3aXwSyVe19v5nnwm6m7qB2cp+9Y2978H5OWs9Fk/io/pket64EP/76piaNTcu/X9ZduD7AU3Zs/u+T8CdmWNOfUPMoxv7Dz/aMCpDM+pxBGTaX5nm2boUva+lgnzCKI4iWTUm1a0N+t4CGby6+7wK8hnhNNfi301xH8Poja7CcIB4cwN9ngpcxi03wa2+1AtehkWE0/jNfmLfdROkobGe1fdzM4ZVn2VM1g7KT8RstyOveC+7jEt5y+i6WYFb78eS4VM/UM70Xhyr2TxtXtiWD/o8KjbM7z0FzwtkdpEAh/JihGcH3kxNM5yI6dXi/sXsBtFkypPXByvYilBI+yVQf6/tm1VTXmBvVJ1t8MfgvTqT9EGZrSVOf1eq29prinXqlwjsaFK78+BQam2aVWNV+JavyM3a10qtgQjVbIOn83KZNl4ScqkS51XZukj6d2Acbbsln5e41T9CZBbloqhNiBf4l6dL6+UgDKHpZ91LI6CMeWisakh6lUfCknmpHs/OYtbKraG21LpuKPtqMJAWt+atvk0Dr7b3yUP5fi2hRLwJe4lr3/95DFXOolqUahO+deilG0wXmJig3PUpdgW0eFVnF69aPBCPfVbuafX6qXbC4/lvDwIAAAAwAAAAAgAAACsAAAACAAAAQCAAAAIAAAAAAAEA");
    //let mut board = BlueprintParser::default().parse("KLUv/WBII80ZAIQCAABj8v//KjVB/y5HXU04Pv9meI7/ogAw2f/GY/9iiv+mAP//rnRiioHlqNEHUAABBUgZZSG8ChJAIwABKgACUAAEEAABEgCBEUBgQookmAAAQwWvVR5VKbgs8lldGuB36k6/hUOWTdMFlKx9diyP+alvPXSZyEUzN+eeuHkOv42JWb2vPGu4xJ+1wk78dZstTyjw9+k8RPMyuJV/mf3lkL5ZfMvgbBMXxLcbIPvUvjgqz8p3/4DiyW2a/iD0Bx88TD4fum63+Kadr9Gn/edh5Ffr+8HcWqcEh74ZJrHT+b9P83tO7lsTF6YedTe3v2oZ+Zcy8rvzY49n+QRXQ+3Ib2q3NQKcWyffNJjfDK0WezH80t80Vllohr4YG6YPt0ggP3j/oa+XsyuTNiFOHM5/aebyZ/sZbqO+NRQWcurHkP4SXsjLeVP81Cz6J8wa/VXzORM3idEx2f2bK/qb/ESQCfivncZJztXorzoHF6aDDcU10wMDw6wpeddTRBj1oTf+9LZ0Bvr6lIVbdwff5aqP05udVH/bsGT3VR6A8FbLK/0lbm0hoFHfGousPJMbuT9xXswz/ccbiRh9tD+jF0x9W6w339mw68zgC+J3/w3NoY+HKxa+4o+0BXuQICM/+q9VmWxlMy9G/ar/1wpux3y/now7+rQzntFpCoPclu9PwG0RPw6umx3w4VPIHXul/eHOfSG+ZSu7AxzYQxo08OyZfwS+ybbHh2m+1W3MtS/tsye7YaQe4qZcNX7og/FXn/rijw/cKC/46F6X4oZ372hOxlX1Fx/6XKi3+en8SEHzJyjfUTGBflU5pP/5rNevhbyM19gJrsUtWVb7Kzo9eJx9LRvVaSti2+6/y4rLFzCyd9i2rzCuNnKs3K/lSgvnO4oRWMdU3n6EtsUg8v+jE+FXEgLaD/Q5SW3DWarDbQtvOs+h2o+6d7vSts+9vE+R0X6gv2OzZTvM4/9//cjvbW539pG3eXVHR40XseSq+p5s8/utcxHd6YnoJn6+P/iauBtyjV9xgUn2bPycZdFypd1srPzqC5+pfl3SPO7Bb35Ph+WAlzvClO7UHThEprHtFHoEaTL9s9z9AwIAAAArAAAAAgAAADYAAAACAAAASCQAAAIAAAAAAAEA");
    //let mut board = BlueprintParser::default().parse("KLUv/WCYAy0EAAQCAABNOD7/ADDZ//9meI7/Yor/LkddrnT//yo1Qf/GY/8wIAADqiAbACUUhMJ4eLDCGWEWBg9XCij40qEFn8B9GFnhAsPPABCAr0XjwhXIsFqYEMEKAwADHxxMXQyGWDCC4oCDHOAgIsBBDnLQCOJgwcCC8oQrBycVEx1QcI4gqH5RAAsCAAAADgAAAAIAAAAVAAAAAgAAAJgEAAACAAAAAAABAA==");
    //let mut board = BlueprintParser::default().parse("KLUv/WA0B7UDAGQCAABNOD7/AC5HXf+S/2P//2KKOE1H/2P/n//GogBj8v8w2a50pgAnIEADohEcKSoYwniQsZIahhAejqmsAgvhcAz3IvibY4UcEECAQAIJEEAgARNYEQJSI4XmCNA3BiKUhRqacHMCAiA0X2C2ck8UYn3SjgUCAAAAFQAAAAIAAAAZAAAAAgAAADQIAAACAAAAAAABAA==");
    let mut board = BlueprintParser::default().parse("KLUv/SAwPQEA2AAAAABNOD7/AAAAAC5HXf+udP//OE1HY/+f/wQAQ8mAADcJWFCYAgAAAAQAAAACAAAAAwAAAAIAAAAwAAAAAgAAAAAAAQA=");
    
    board.print();
    board.update();
    board.print();
    //board.update();
    board.print();



    let a = board.make_state_vec();

    println!("{a:?}");

    //loop {
    //    print!("\x1B[0;0H");
    //    board.print();
    //    board.update();
    //    let mut child = Command::new("sleep").arg("0.1").spawn().unwrap();
    //    let _result = child.wait().unwrap();
    //}
}
