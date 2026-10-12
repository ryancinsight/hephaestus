use super::super::{AdaGrad, Adam, AdamW, RmsProp, Sgd, StatefulUpdateRule};
use crate::domain::dialect::Host;

#[test]
fn f64_bodies_exist_exactly_where_f32_literals_appear() {
    // Adam-family bodies pin `1.0f`; their f64 spellings use `1.0`
    // (AbstractFloat in WGSL, double in kernel C) and carry no `f`
    // suffix anywhere.
    for body in [
        <Adam as StatefulUpdateRule<Host>>::BODY_F64,
        <AdamW as StatefulUpdateRule<Host>>::BODY_F64,
        <RmsProp as StatefulUpdateRule<Host>>::BODY_F64,
    ] {
        let body = body.expect("f32-literal rule needs a BODY_F64");
        assert!(body.contains("1.0"), "f64 body must spell unit literals");
        assert!(
            !body.contains("1.0f"),
            "f64 body must not carry f32-suffixed literals"
        );
    }
    // SGD and AdaGrad bodies are operand-only arithmetic; backends
    // reuse BODY verbatim.
    assert!(<Sgd as StatefulUpdateRule<Host>>::BODY_F64.is_none());
    assert!(<AdaGrad as StatefulUpdateRule<Host>>::BODY_F64.is_none());
    assert!(!<Sgd as StatefulUpdateRule<Host>>::BODY.contains("1.0f"));
    assert!(!<AdaGrad as StatefulUpdateRule<Host>>::BODY.contains("1.0f"));
}

#[test]
fn width_selectors_resolve_both_spellings() {
    // f32 resolves to the f32-only items for every rule.
    assert_eq!(
        <Adam as StatefulUpdateRule<Host>>::body_for::<f32>(),
        <Adam as StatefulUpdateRule<Host>>::BODY
    );
    assert_eq!(
        <Adam as StatefulUpdateRule<Host>>::parameter_fields_for::<f32>(),
        <Adam as StatefulUpdateRule<Host>>::PARAMETER_FIELDS
    );
    // f64 resolves to the f64 spelling where one exists, else BODY.
    assert_eq!(
        <Adam as StatefulUpdateRule<Host>>::body_for::<f64>(),
        <Adam as StatefulUpdateRule<Host>>::BODY_F64.expect("Adam ships BODY_F64")
    );
    assert_eq!(
        <Sgd as StatefulUpdateRule<Host>>::body_for::<f64>(),
        <Sgd as StatefulUpdateRule<Host>>::BODY
    );
    assert_eq!(
        <Sgd as StatefulUpdateRule<Host>>::parameter_fields_for::<f64>(),
        <Sgd as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64
    );
}

#[test]
fn f64_field_lists_drop_only_the_padding() {
    assert_eq!(
        <Sgd as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64,
        &["learning_rate", "momentum"]
    );
    assert_eq!(
        <Adam as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64.len(),
        <Adam as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len() - 2
    );
    assert_eq!(
        <AdamW as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64.len(),
        <AdamW as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len() - 1
    );
    assert_eq!(
        <RmsProp as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64.len(),
        <RmsProp as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len() - 1
    );
    assert_eq!(
        <AdaGrad as StatefulUpdateRule<Host>>::PARAMETER_FIELDS_F64.len(),
        <AdaGrad as StatefulUpdateRule<Host>>::PARAMETER_FIELDS.len() - 2
    );
}
