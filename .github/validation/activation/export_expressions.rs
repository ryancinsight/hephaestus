use hephaestus_core::*;
fn main() {
    println!("ErfcOp\tWgsl\t{:?}", <ErfcOp as UnaryExpr<Wgsl>>::EXPR);
    println!("ErfcOp\tCudaC\t{:?}", <ErfcOp as UnaryExpr<CudaC>>::EXPR);
    println!("ErfcOp\tHipC\t{:?}", <ErfcOp as UnaryExpr<HipC>>::EXPR);
    println!("GeluOp\tWgsl\t{:?}", <GeluOp as UnaryExpr<Wgsl>>::EXPR);
    println!("GeluOp\tCudaC\t{:?}", <GeluOp as UnaryExpr<CudaC>>::EXPR);
    println!("GeluOp\tHipC\t{:?}", <GeluOp as UnaryExpr<HipC>>::EXPR);
    println!(
        "GeluGradOp\tWgsl\t{:?}",
        <GeluGradOp as UnaryExpr<Wgsl>>::EXPR
    );
    println!(
        "GeluGradOp\tCudaC\t{:?}",
        <GeluGradOp as UnaryExpr<CudaC>>::EXPR
    );
    println!(
        "GeluGradOp\tHipC\t{:?}",
        <GeluGradOp as UnaryExpr<HipC>>::EXPR
    );
    println!(
        "GeluTanhGradOp\tWgsl\t{:?}",
        <GeluTanhGradOp as UnaryExpr<Wgsl>>::EXPR
    );
    println!(
        "GeluTanhGradOp\tCudaC\t{:?}",
        <GeluTanhGradOp as UnaryExpr<CudaC>>::EXPR
    );
    println!(
        "GeluTanhGradOp\tHipC\t{:?}",
        <GeluTanhGradOp as UnaryExpr<HipC>>::EXPR
    );
    println!(
        "SoftplusOp\tWgsl\t{:?}",
        <SoftplusOp as UnaryExpr<Wgsl>>::EXPR
    );
    println!(
        "SoftplusOp\tCudaC\t{:?}",
        <SoftplusOp as UnaryExpr<CudaC>>::EXPR
    );
    println!(
        "SoftplusOp\tHipC\t{:?}",
        <SoftplusOp as UnaryExpr<HipC>>::EXPR
    );
    println!("SiluOp\tWgsl\t{:?}", <SiluOp as UnaryExpr<Wgsl>>::EXPR);
    println!("SiluOp\tCudaC\t{:?}", <SiluOp as UnaryExpr<CudaC>>::EXPR);
    println!("SiluOp\tHipC\t{:?}", <SiluOp as UnaryExpr<HipC>>::EXPR);
    println!("MishOp\tWgsl\t{:?}", <MishOp as UnaryExpr<Wgsl>>::EXPR);
    println!("MishOp\tCudaC\t{:?}", <MishOp as UnaryExpr<CudaC>>::EXPR);
    println!("MishOp\tHipC\t{:?}", <MishOp as UnaryExpr<HipC>>::EXPR);
    println!(
        "MishGradOp\tWgsl\t{:?}",
        <MishGradOp as UnaryExpr<Wgsl>>::EXPR
    );
    println!(
        "MishGradOp\tCudaC\t{:?}",
        <MishGradOp as UnaryExpr<CudaC>>::EXPR
    );
    println!(
        "MishGradOp\tHipC\t{:?}",
        <MishGradOp as UnaryExpr<HipC>>::EXPR
    );
}
