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
//! | [`presets`] | consumers: the built-in PySCF level, preset files, PySCF check |
//! | [`algorithms`] | radial/angular rules, pruning and partition formulas |
//! | [`partition_derivatives`] | tensor layouts and the streaming derivative API |
//! | [`becke_derivatives`] | derivation of the classic Becke gradient and Hessian |
//! | [`scheme_variants`] | SSF, SS, LKO and mBecke formula and cost differences |
//! | [`performance`] | measured throughput and build-target guidance |

// Markdown runs before KaTeX: an `_` that does not follow a letter or digit is
// written `\_`; commands spelled as backslash + punctuation (`\,`, `\{`) use
// named forms (`\thinspace`, `\lbrace`, `\cr`).  `tests/math_markdown.rs`
// enforces both rules.

pub mod algorithms {
    #![doc = include_str!("../../docs/algorithms.md")]
}

pub mod becke_derivatives {
    #![doc = include_str!("../../docs/becke-derivatives.md")]
}

pub mod guide {
    #![doc = include_str!("../../docs/user-guide.md")]
}

pub mod presets {
    #![doc = include_str!("../../docs/presets.md")]
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
