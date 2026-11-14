//! Long-form documentation.
//!
//! The reference pages below are part of the crate documentation and are
//! also available offline in the repository. Start with the
//! [guide] if you want to build a grid, or with the [algorithms] page if you
//! want the formulas.
//!
//! | Page | Audience |
//! |---|---|
//! | [`guide`] | consumers: geometry -> grid -> derivatives -> ordering |
//! | [`algorithms`] | radial/angular rules, pruning and partition formulas |
//! | [`partition_derivatives`] | tensor layouts and the streaming derivative API |
//! | [`becke_derivatives`] | derivation of the classic Becke gradient and Hessian |
//! | [`scheme_variants`] | SSF, SS, LKO and mBecke formula and cost differences |
//! | [`performance`] | measured throughput and build-target guidance |

// The Markdown sources write `_` inside math as `&#95;`: rustdoc parses
// doc comments as Markdown before KaTeX sees them, and a bare pair of
// underscores would otherwise split a formula into emphasis runs.

pub mod algorithms {
    #![doc = include_str!("../../docs/algorithms.md")]
}

pub mod becke_derivatives {
    #![doc = include_str!("../../docs/becke-derivatives.md")]
}

pub mod guide {
    #![doc = include_str!("../../docs/user-guide.md")]
}

pub mod partition_derivatives {
    #![doc = include_str!("../../docs/partition-derivatives.md")]
}

pub mod performance {
    #![doc = include_str!("../../docs/performance.md")]
}

pub mod scheme_variants {
    #![doc = include_str!("../../docs/scheme-variants.md")]
}
