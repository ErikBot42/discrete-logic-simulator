#![no_main]

use libfuzzer_sys::fuzz_target;

use logic_simulator::blueprint::fuzz_vcb_board_creation;
use logic_simulator::blueprint::parse::ArbitraryVcbPlainBoard;

use libfuzzer_sys::arbitrary::{Arbitrary, Result, Unstructured};

#[cfg(fuzzing)]
#[derive(Debug)]
struct Foo {
    a: u64,
    b: u64,
}

impl<'a> Arbitrary<'a> for Foo {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let a = u64::arbitrary(u)?;
        let b = u64::arbitrary(u)?;
        Ok(Foo { a, b })
    }
}

fuzz_target!(|data: ArbitraryVcbPlainBoard| {
    let _ = fuzz_vcb_board_creation(data);  
});
