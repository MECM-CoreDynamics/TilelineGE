use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    tl_core::gms::render_benchmark::run_from_env()
}
