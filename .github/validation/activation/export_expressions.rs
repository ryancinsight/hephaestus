use hephaestus_core::*;
use std::io::{self, LineWriter, Write};
use std::process::ExitCode;
fn protocol(output: &mut impl Write) -> io::Result<()> {
    writeln!(
        output,
        "ErfcOp\tWgsl\t{:?}",
        <ErfcOp as UnaryExpr<Wgsl>>::EXPR
    )?;
    writeln!(
        output,
        "ErfcOp\tCudaC\t{:?}",
        <ErfcOp as UnaryExpr<CudaC>>::EXPR
    )?;
    writeln!(
        output,
        "ErfcOp\tHipC\t{:?}",
        <ErfcOp as UnaryExpr<HipC>>::EXPR
    )?;
    writeln!(
        output,
        "GeluOp\tWgsl\t{:?}",
        <GeluOp as UnaryExpr<Wgsl>>::EXPR
    )?;
    writeln!(
        output,
        "GeluOp\tCudaC\t{:?}",
        <GeluOp as UnaryExpr<CudaC>>::EXPR
    )?;
    writeln!(
        output,
        "GeluOp\tHipC\t{:?}",
        <GeluOp as UnaryExpr<HipC>>::EXPR
    )?;
    writeln!(
        output,
        "GeluGradOp\tWgsl\t{:?}",
        <GeluGradOp as UnaryExpr<Wgsl>>::EXPR
    )?;
    writeln!(
        output,
        "GeluGradOp\tCudaC\t{:?}",
        <GeluGradOp as UnaryExpr<CudaC>>::EXPR
    )?;
    writeln!(
        output,
        "GeluGradOp\tHipC\t{:?}",
        <GeluGradOp as UnaryExpr<HipC>>::EXPR
    )?;
    writeln!(
        output,
        "GeluTanhGradOp\tWgsl\t{:?}",
        <GeluTanhGradOp as UnaryExpr<Wgsl>>::EXPR
    )?;
    writeln!(
        output,
        "GeluTanhGradOp\tCudaC\t{:?}",
        <GeluTanhGradOp as UnaryExpr<CudaC>>::EXPR
    )?;
    writeln!(
        output,
        "GeluTanhGradOp\tHipC\t{:?}",
        <GeluTanhGradOp as UnaryExpr<HipC>>::EXPR
    )?;
    writeln!(
        output,
        "SoftplusOp\tWgsl\t{:?}",
        <SoftplusOp as UnaryExpr<Wgsl>>::EXPR
    )?;
    writeln!(
        output,
        "SoftplusOp\tCudaC\t{:?}",
        <SoftplusOp as UnaryExpr<CudaC>>::EXPR
    )?;
    writeln!(
        output,
        "SoftplusOp\tHipC\t{:?}",
        <SoftplusOp as UnaryExpr<HipC>>::EXPR
    )?;
    writeln!(
        output,
        "SiluOp\tWgsl\t{:?}",
        <SiluOp as UnaryExpr<Wgsl>>::EXPR
    )?;
    writeln!(
        output,
        "SiluOp\tCudaC\t{:?}",
        <SiluOp as UnaryExpr<CudaC>>::EXPR
    )?;
    writeln!(
        output,
        "SiluOp\tHipC\t{:?}",
        <SiluOp as UnaryExpr<HipC>>::EXPR
    )?;
    writeln!(
        output,
        "MishOp\tWgsl\t{:?}",
        <MishOp as UnaryExpr<Wgsl>>::EXPR
    )?;
    writeln!(
        output,
        "MishOp\tCudaC\t{:?}",
        <MishOp as UnaryExpr<CudaC>>::EXPR
    )?;
    writeln!(
        output,
        "MishOp\tHipC\t{:?}",
        <MishOp as UnaryExpr<HipC>>::EXPR
    )?;
    writeln!(
        output,
        "MishGradOp\tWgsl\t{:?}",
        <MishGradOp as UnaryExpr<Wgsl>>::EXPR
    )?;
    writeln!(
        output,
        "MishGradOp\tCudaC\t{:?}",
        <MishGradOp as UnaryExpr<CudaC>>::EXPR
    )?;
    writeln!(
        output,
        "MishGradOp\tHipC\t{:?}",
        <MishGradOp as UnaryExpr<HipC>>::EXPR
    )?;
    Ok(())
}

fn main() -> ExitCode {
    let mut output = LineWriter::new(io::stdout());
    let outcome = protocol(&mut output);
    let output_flush = output.flush();
    match outcome.and(output_flush) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let mut errors = LineWriter::new(io::stderr());
            let _ = writeln!(errors, "PROTOCOL_IO_FAILURE\t{error:?}");
            let _ = errors.flush();
            ExitCode::from(3)
        }
    }
}
