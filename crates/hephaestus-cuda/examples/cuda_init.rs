fn main() {
    println!("Loading nvcuda.dll dynamically...");
    // Let's try to load the cuda-async library
    println!("Initializing cuda_async...");
    let dev_res = cuda_async::device_context::with_device(0, |device| {
        println!("Successfully acquired device inside closure!");
        device.clone()
    });
    match dev_res {
        Ok(dev) => {
            println!("Acquired device: {:?}", dev);
            println!("Binding to thread...");
            match dev.bind_to_thread() {
                Ok(_) => println!("Successfully bound to thread!"),
                Err(e) => println!("Error binding: {:?}", e),
            }
        }
        Err(e) => {
            println!("Failed to acquire device: {:?}", e);
        }
    }
}
