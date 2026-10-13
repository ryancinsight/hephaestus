use super::super::*;

#[test]
fn consumer_defined_op_composes_with_the_vocabulary() {
    // A consumer-side fused op: implement the expression trait for a
    // local ZST in the targeted dialect — no substrate changes needed.
    #[derive(Clone, Copy, Debug, Default)]
    struct AffineClampOp;
    impl UnaryExpr<Wgsl> for AffineClampOp {
        const EXPR: &'static str = "clamp(x * 2.0 + 1.0, 0.0, 10.0)";
    }
    fn expr_of<Op: UnaryExpr<Wgsl>>() -> &'static str {
        Op::EXPR
    }
    assert_eq!(
        expr_of::<AffineClampOp>(),
        "clamp(x * 2.0 + 1.0, 0.0, 10.0)"
    );
}
