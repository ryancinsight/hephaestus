use libloading::{Library, Symbol};

fn main() {
    println!("Loading nvcuda.dll...");
    let lib = unsafe { Library::new("nvcuda.dll") };
    match lib {
        Ok(l) => {
            println!("nvcuda.dll loaded successfully!");
            println!("Looking up cuInit...");
            unsafe {
                let cu_init: Result<Symbol<unsafe extern "C" fn(u32) -> i32>, _> = l.get(b"cuInit");
                match cu_init {
                    Ok(f) => {
                        println!("cuInit found! Calling cuInit(0)...");
                        let res = f(0);
                        println!("cuInit(0) returned: {}", res);
                    }
                    Err(e) => {
                        println!("Failed to get cuInit: {:?}", e);
                    }
                }
            }
        }
        Err(e) => {
            println!("Failed to load nvcuda.dll: {:?}", e);
        }
    }
}
