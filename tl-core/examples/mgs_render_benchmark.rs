use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    tl_core::mgs::render_benchmark::run_from_env()
}
