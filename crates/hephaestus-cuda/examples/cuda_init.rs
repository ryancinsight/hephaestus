fn main() {
    println!("Initializing cuda-oxide driver substrate...");
    if let Err(error) = cuda_oxide::Cuda::init() {
        println!("Failed to initialize CUDA driver: {error}");
        return;
    }

    match cuda_oxide::Cuda::list_devices() {
        Ok(devices) => {
            println!("CUDA devices: {}", devices.len());
            for (ordinal, device) in devices.iter().enumerate() {
                let name = device.name().unwrap_or_else(|error| format!("{error}"));
                println!("{ordinal}: {name}");
            }
        }
        Err(error) => println!("Failed to enumerate CUDA devices: {error}"),
    }
}
