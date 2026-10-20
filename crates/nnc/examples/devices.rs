//! Print the devices nnc can use.
fn main() {
    println!("GPUs visible to nnc: {}", nnc::gpu_count());
    println!("default device: {:?}", nnc::default_device());
}
