//! Radial quadrature grids.
//!
//! The module is split by function:
//!
//! * [`kernels`] — every radial node mapping, ORCA, PySCF and
//!   Euler–Maclaurin alike, as `(n, xi)` kernels;
//!
//! [`Radial`] is the unified selector for all eight built-in constructions.
//! Every construction is a function of the shell count `n` and a radial length
//! scale $\xi$ (Bohr); [`kernels`] holds the node mappings.  The scale is supplied
//! explicitly ([`Radial::radial_rd`], [`Radial::grid_xi`]), from an element
//! preset ([`Radial::xi_for_element`], [`Radial::grid_for_element`]), or
//! through an explicit [`RadialScale`] accepted by [`Radial::grid`].
//!
//! For a custom construction, implement [`RadialMap`] for the node mapping and
//! pair it with a [`RadialScalePolicy`] in a [`RadialSpec`]. This keeps the
//! map independent of the element-to-$\xi$ rule; a map with no element rule
//! can use [`RadialScalePolicy::Explicit`] and an explicit $\xi$. The built-in
//! presets are available through [`Radial::spec`].
//!
//! [`shells`] additionally applies a radial cutoff and records the
//! dimensionless ratio $r / \xi$; it is available to every caller that needs
//! cutoff-limited radial shells.
//!
//! There is a single weight convention: [`Radial::grid_xi`] / [`Radial::grid`]
//! return the PySCF volume element $4\pi r^2\ \Delta r$, to be paired with the
//! sum-to-one angular weights of [`crate::angular::angular_grid`].  The
//! product $w_\mathrm{rad} w_\mathrm{ang}$ is the volume element, so the
//! placement of $4\pi$ is invisible to the integrator.

pub mod kernels;

pub use kernels::*;

use crate::error::{Error, Result};
use crate::params::treutler_xi;

/// Radial quadrature construction (unified selector).
///
/// The PySCF variants port `pyscf.dft.radi`; `OrcaOptM3`/`OrcaM3` are the
/// two Chebyshev-style maps used by ORCA, and `EulerMaclaurin` is the
/// squared-ratio map.  A variant selects the node mapping only: the
/// element-to-scale rule is a separate step ([`Radial::xi_for_element`]).
/// The ORCA and Euler–Maclaurin maps have no element table in this crate;
/// their scale comes from a [`crate::preset::GridPreset`] or an explicit
/// [`RadialScale::Xi`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Radial {
    /// PySCF `radi.treutler_ahlrichs` (M4; PySCF `Grids.radi_method`
    /// default).
    #[default]
    TreutlerAhlrichs,
    /// PySCF `radi.delley` (log2 / Gauss–Legendre).
    Delley,
    /// PySCF `radi.mura_knowles` (log3).
    MuraKnowles,
    /// PySCF `radi.gauss_chebyshev` (M3; `gen_atomic_grids` default
    /// argument).
    GaussChebyshev,
    /// PySCF `radi.becke` (Becke's mapping, element-scaled by Bragg radii).
    Becke,
    /// The exact ORCA 6.1 "OptM3 with GC (2021)" grid (the ORCA default).
    OrcaOptM3,
    /// The published plain Chebyshev M3 reference grid.
    OrcaM3,
    /// The Euler–Maclaurin squared-ratio map
    /// $r_i = \xi\left((i+1)/(n_\mathrm{rad}-i)\right)^2$.
    ///
    /// The scale is supplied by the caller; the crate has no per-element
    /// table for this map.
    EulerMaclaurin,
}

/// How a radial grid chooses its length scale $\xi$ (Bohr).
///
/// This is the single-input form of the "element vs explicit $\xi$" choice:
/// [`RadialScale::Z`] resolves through [`Radial::xi_for_element`], while
/// [`RadialScale::Xi`] is used verbatim.  [`Radial::grid`] takes the enum
/// explicitly; [`Radial::grid_for_element`] is the element-preset shorthand.
///
/// ```
/// use ruaa_dft_grids::radial::{Radial, RadialScale};
///
/// let a = Radial::TreutlerAhlrichs.grid(20, RadialScale::Z(6)).unwrap();
/// let b = Radial::TreutlerAhlrichs.grid_for_element(20, 6).unwrap();
/// assert_eq!(a, b);
///
/// let c = Radial::TreutlerAhlrichs.grid(20, RadialScale::Xi(1.1)).unwrap();
/// let d = Radial::TreutlerAhlrichs.grid_xi(20, 1.1);
/// assert_eq!(c, d);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RadialScale {
    /// Use the element's preset: `Radial::xi_for_element(z)`.
    Z(u32),
    /// Use this explicit $\xi$ (Bohr).
    Xi(f64),
}

impl RadialScale {
    /// The numeric $\xi$ this scale denotes for `method`.
    ///
    /// # Errors
    /// Propagates [`Radial::xi_for_element`] when the scale is
    /// [`Self::Z`] and the element has no table entry.
    pub fn resolve(self, method: Radial) -> Result<f64> {
        self.resolve_policy(method.scale_policy())
    }

    /// The numeric $\xi$ this scale denotes for a scale policy.
    ///
    /// # Errors
    /// Propagates [`RadialScalePolicy::xi_for_element`] when the scale is
    /// [`Self::Z`] and the policy has no element rule or no entry for it.
    pub fn resolve_policy(self, policy: RadialScalePolicy) -> Result<f64> {
        match self {
            Self::Z(z) => policy.xi_for_element(z),
            Self::Xi(xi) => Ok(xi),
        }
    }
}

/// Element-to-$\xi$ rule used by a radial construction.
///
/// This is the scale half of a radial specification. The node mapping is a
/// separate [`RadialMap`]; [`Radial`] remains the convenience enum that pairs
/// a built-in map with its usual scale rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RadialScalePolicy {
    /// PySCF `radi._treutler_ahlrichs_xi[z]`.
    PyScfTreutler,
    /// PySCF `radi.delley` outer radius (12 Bohr).
    Delley,
    /// PySCF `radi.mura_knowles` far distance.
    MuraKnowles,
    /// Unit scale used by PySCF `radi.gauss_chebyshev`.
    Unit,
    /// PySCF `radi.BRAGG_RADII` with the PySCF Becke halving rule.
    PyScfBecke,
    /// No element rule: the caller must supply an explicit $\xi$.
    Explicit,
}

impl RadialScalePolicy {
    /// The length scale $\xi$ (Bohr) for element `z`.
    ///
    /// # Errors
    /// [`Error::MissingElementData`] when the backing table has no entry, and
    /// [`Error::UnsupportedGrid`] for [`Self::Explicit`].
    pub fn xi_for_element(self, z: u32) -> Result<f64> {
        match self {
            Self::PyScfTreutler => treutler_xi(z).ok_or(Error::MissingElementData {
                z,
                table: "PySCF Treutler xi",
            }),
            Self::Delley => Ok(kernels::DELLEY_R_OUTER),
            Self::MuraKnowles => Ok(kernels::mura_knowles_far(z)),
            Self::Unit => Ok(kernels::GAUSS_CHEBYSHEV_SCALE),
            Self::PyScfBecke => kernels::becke_rm(z).ok_or(Error::MissingElementData {
                z,
                table: "PySCF BRAGG",
            }),
            Self::Explicit => Err(Error::UnsupportedGrid {
                name: "radial scale".to_string(),
                reason: "this policy has no element rule; supply an explicit xi".to_string(),
            }),
        }
    }
}

impl Radial {
    /// The element-to-$\xi$ policy paired with this built-in map.
    #[must_use]
    pub fn scale_policy(self) -> RadialScalePolicy {
        match self {
            Self::TreutlerAhlrichs => RadialScalePolicy::PyScfTreutler,
            Self::Delley => RadialScalePolicy::Delley,
            Self::MuraKnowles => RadialScalePolicy::MuraKnowles,
            Self::GaussChebyshev => RadialScalePolicy::Unit,
            Self::Becke => RadialScalePolicy::PyScfBecke,
            Self::OrcaOptM3 | Self::OrcaM3 | Self::EulerMaclaurin => RadialScalePolicy::Explicit,
        }
    }

    /// The radial length scale $\xi$ (Bohr) this construction's backend uses for
    /// element `z`.
    ///
    /// * PySCF methods use their own table/rule: Treutler
    ///   `_treutler_ahlrichs_xi[z]`; Becke `BRAGG[z]` (halved for every `z`
    ///   except hydrogen); Mura–Knowles `far` (7 Bohr for Li, Be, Na, Mg, K,
    ///   Ca; else 5.2); Delley `r_outer = 12`; Gauss–Chebyshev unit scale 1.
    /// * `OrcaOptM3`, `OrcaM3` and `EulerMaclaurin` have no element table in
    ///   this crate. Their scale comes from the caller (the `radial_scale`
    ///   field of a [`crate::preset::ElementPreset`], or
    ///   [`RadialScale::Xi`]).
    ///
    /// # Errors
    /// [`Error::MissingElementData`] if the backing PySCF table has no entry
    /// for `z` (PySCF Treutler: Z > 103; PySCF Becke: Z outside
    /// `BRAGG_BOHR`), and [`Error::UnsupportedGrid`] for a map with no
    /// element rule.
    pub fn xi_for_element(self, z: u32) -> Result<f64> {
        self.scale_policy().xi_for_element(z)
    }

    /// Raw radial nodes and increments `(r_i, dr_i)` for an explicit scale
    /// $\xi$ (Bohr).
    ///
    /// This is the raw node/increment kernel behind [`Radial::grid_xi`]; the
    /// caller forms the volume element.  For the ORCA variants `dr` is
    /// recovered from the ORCA weight $w = r^2\ \Delta r$, so it is
    /// mathematically, not bit-for-bit, the derivative computed inside
    /// [`kernels`].
    #[must_use]
    pub fn radial_rd(self, n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
        match self {
            Self::TreutlerAhlrichs => kernels::treutler_ahlrichs_xi(n, xi),
            Self::Delley => kernels::delley_xi(n, xi),
            Self::MuraKnowles => kernels::mura_knowles_xi(n, xi),
            Self::GaussChebyshev => kernels::gauss_chebyshev_xi(n, xi),
            Self::Becke => kernels::becke_xi(n, xi),
            Self::OrcaOptM3 => orca_rd(kernels::radial_grid_orca, n, xi),
            Self::OrcaM3 => orca_rd(kernels::radial_grid_m3, n, xi),
            Self::EulerMaclaurin => kernels::euler_maclaurin_xi(n, xi),
        }
    }

    /// Radial volume element $4\pi r^2\ \Delta r$ for an explicit scale $\xi$.
    ///
    /// Pair with [`crate::angular::angular_grid`] (weights sum to one) to
    /// form the quadrature weight `w_rad * w_ang`.  Available for every
    /// [`Radial`] variant, including the ORCA kernels (whose weight
    /// already carries the `r^2` measure; it is scaled by `4*pi` here).
    #[must_use]
    pub fn grid_xi(self, n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
        match self {
            Self::OrcaOptM3 => {
                let (r, w) = kernels::radial_grid_orca(n, xi);
                (r, w.into_iter().map(|w| (4.0 * crate::PI) * w).collect())
            }
            Self::OrcaM3 => {
                let (r, w) = kernels::radial_grid_m3(n, xi);
                (r, w.into_iter().map(|w| (4.0 * crate::PI) * w).collect())
            }
            _ => {
                let (r, dr) = self.radial_rd(n, xi);
                let w = r
                    .iter()
                    .zip(&dr)
                    .map(|(&r, &dr)| (4.0 * crate::PI) * (r * r) * dr)
                    .collect();
                (r, w)
            }
        }
    }

    /// Radial volume element $4\pi r^2\ \Delta r$ for an explicit
    /// [`RadialScale`].
    ///
    /// Use [`Self::grid_for_element`] for the common element-preset case, or
    /// [`Self::grid_xi`] when the scale is already known.
    ///
    /// ```
    /// use ruaa_dft_grids::radial::{Radial, RadialScale};
    ///
    /// # fn main() -> ruaa_dft_grids::Result<()> {
    /// let (r, w) = Radial::TreutlerAhlrichs.grid(20, RadialScale::Z(6))?;
    /// assert_eq!(r.len(), 20);
    /// assert!(w.iter().all(|w| *w >= 0.0));
    ///
    /// let (r, w) = Radial::TreutlerAhlrichs.grid(20, RadialScale::Xi(1.1))?;
    /// assert_eq!(r.len(), 20);
    /// assert!(w.iter().all(|w| *w >= 0.0));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] if `n == 0` or the resolved scale is not a
    /// finite positive value; if `scale` is [`RadialScale::Z`] and the backing
    /// table has no entry for that element (see [`Radial::xi_for_element`]).
    pub fn grid(self, n: usize, scale: RadialScale) -> Result<(Vec<f64>, Vec<f64>)> {
        let xi = scale.resolve(self)?;
        validate_radial_grid(n, xi)?;
        Ok(self.grid_xi(n, xi))
    }

    /// Radial volume element $4\pi r^2\ \Delta r$ for element `z`.
    ///
    /// # Errors
    /// The errors of [`Self::grid`] with [`RadialScale::Z`].
    pub fn grid_for_element(self, n: usize, z: u32) -> Result<(Vec<f64>, Vec<f64>)> {
        self.grid(n, RadialScale::Z(z))
    }

    /// This built-in map paired with its usual element-scale policy.
    #[must_use]
    pub fn spec(self) -> RadialSpec<Radial> {
        RadialSpec::new(self, self.scale_policy())
    }

    /// Human-readable name, useful for logs/asset writers.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::TreutlerAhlrichs => "treutler",
            Self::Delley => "delley",
            Self::MuraKnowles => "mura_knowles",
            Self::GaussChebyshev => "gauss_chebyshev",
            Self::Becke => "becke",
            Self::OrcaOptM3 => "orca_opt_m3",
            Self::OrcaM3 => "orca_m3",
            Self::EulerMaclaurin => "euler_maclaurin",
        }
    }
}

/// A radial node mapping, independent of its element-scale policy.
///
/// Implement this trait to add a map without choosing an element-to-$\xi$
/// rule. Combine it with a [`RadialScalePolicy`] in a [`RadialSpec`]; a map
/// with no element rule can use [`RadialScalePolicy::Explicit`] and be
/// evaluated with an explicit $\xi$ through [`RadialSpec::grid_xi`].  The
/// standard grid specs and builders accept only the built-in [`Radial`]
/// selector; a custom map is combined with the assembled
/// [`crate::atomgrid::AtomicGrid`] path.
///
/// `radial_rd` returns `(r, dr)`; the default `grid_xi` forms the
/// $4\pi r^2\thinspace dr$ volume element. A map whose kernel returns `(r, r^2 dr)`
/// directly must override `grid_xi` (the built-in ORCA maps do).
pub trait RadialMap {
    /// Raw radial nodes and increments `(r_i, dr_i)` for an explicit scale
    /// $\xi$ (Bohr).
    fn radial_rd(&self, n: usize, xi: f64) -> (Vec<f64>, Vec<f64>);

    /// Radial volume element $4\pi r^2\ \Delta r$ for an explicit scale.
    fn grid_xi(&self, n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
        let (r, dr) = self.radial_rd(n, xi);
        let w = r
            .iter()
            .zip(&dr)
            .map(|(&r, &dr)| (4.0 * crate::PI) * (r * r) * dr)
            .collect();
        (r, w)
    }

    /// Human-readable map name.
    fn name(&self) -> &'static str;
}

impl RadialMap for Radial {
    fn radial_rd(&self, n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
        Radial::radial_rd(*self, n, xi)
    }

    fn grid_xi(&self, n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
        Radial::grid_xi(*self, n, xi)
    }

    fn name(&self) -> &'static str {
        Radial::name(*self)
    }
}

/// A radial map plus its element-scale policy.
///
/// This is the extension-friendly form of [`Radial`]: the map determines the
/// nodes, while the policy determines how a nuclear charge selects $\xi$.
/// The built-in presets are available through [`Radial::spec`].
///
/// ```
/// use ruaa_dft_grids::radial::{Radial, RadialScale, RadialScalePolicy, RadialSpec};
///
/// # fn main() -> ruaa_dft_grids::Result<()> {
/// let spec = RadialSpec::new(Radial::TreutlerAhlrichs, RadialScalePolicy::PyScfTreutler);
/// let (r, w) = spec.grid(20, RadialScale::Z(6))?;
/// assert_eq!(r.len(), 20);
/// assert_eq!(w.len(), 20);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct RadialSpec<M: RadialMap> {
    /// Node mapping.
    pub map: M,
    /// Element-to-$\xi$ policy.
    pub scale: RadialScalePolicy,
}

impl<M: RadialMap> RadialSpec<M> {
    /// Pair a map with a scale policy.
    #[must_use]
    pub fn new(map: M, scale: RadialScalePolicy) -> Self {
        Self { map, scale }
    }

    /// Raw radial nodes and increments `(r_i, dr_i)`.
    #[must_use]
    pub fn radial_rd(&self, n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
        self.map.radial_rd(n, xi)
    }

    /// Radial volume element $4\pi r^2\ \Delta r$.
    #[must_use]
    pub fn grid_xi(&self, n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
        self.map.grid_xi(n, xi)
    }

    /// Human-readable map name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.map.name()
    }

    /// The element-scale rule applied by [`Self::grid`].
    ///
    /// # Errors
    /// [`Error::MissingElementData`] when the policy has no entry for `z`,
    /// and [`Error::UnsupportedGrid`] for [`RadialScalePolicy::Explicit`].
    pub fn xi_for_element(&self, z: u32) -> Result<f64> {
        self.scale.xi_for_element(z)
    }

    /// Radial volume element for an explicit [`RadialScale`].
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] if `n == 0` or the resolved scale is not a
    /// finite positive value; propagates [`Self::xi_for_element`] for an
    /// element scale.
    pub fn grid(&self, n: usize, scale: RadialScale) -> Result<(Vec<f64>, Vec<f64>)> {
        let xi = scale.resolve_policy(self.scale)?;
        validate_radial_grid(n, xi)?;
        Ok(self.grid_xi(n, xi))
    }

    /// Radial volume element for element `z`.
    ///
    /// # Errors
    /// The errors of [`Self::grid`] with [`RadialScale::Z`].
    pub fn grid_for_element(&self, n: usize, z: u32) -> Result<(Vec<f64>, Vec<f64>)> {
        self.grid(n, RadialScale::Z(z))
    }
}

fn validate_radial_grid(n: usize, xi: f64) -> Result<()> {
    if n == 0 {
        return Err(Error::InvalidArgument {
            argument: "radial shell count",
        });
    }
    if !xi.is_finite() || xi <= 0.0 {
        return Err(Error::InvalidArgument {
            argument: "radial scale",
        });
    }
    Ok(())
}

/// Recover `(r, dr)` from an ORCA kernel that returns `(r, r^2 dr)`.
fn orca_rd(
    kernel: fn(usize, f64) -> (Vec<f64>, Vec<f64>),
    n: usize,
    xi: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (r, w) = kernel(n, xi);
    let dr = r
        .iter()
        .zip(&w)
        .map(|(&r, &w)| if r == 0.0 { 0.0 } else { w / (r * r) })
        .collect();
    (r, dr)
}

/// One generated radial shell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialShell {
    /// Node radius, Bohr.
    pub r: f64,
    /// Radial volume element $4\pi r^2\ \Delta r$.
    pub w: f64,
    /// Whether the shell carries points (`r <= cutoff`).
    pub active: bool,
    /// Dimensionless radius $r / \xi$. The five-batch region ladders
    /// ([`crate::prune::LadderPrune`]) compare against this ratio.
    pub ratio: f64,
}

/// Dimensionless node ratios $r / \xi$ for `method`.
///
/// For [`Radial::EulerMaclaurin`] this returns the exact integer ratio
/// $((i+1)/(n-i))^2$; deriving it as $r / \xi$ could move a tie shell across a
/// region boundary through rounding.  Other mappings divide the node radii
/// by the supplied scale.
#[must_use]
pub fn ratios(method: Radial, rads: &[f64], xi: f64) -> Vec<f64> {
    if method == Radial::EulerMaclaurin {
        kernels::euler_maclaurin_ratios(rads.len())
    } else if xi == 0.0 {
        vec![0.0; rads.len()]
    } else {
        rads.iter().map(|&r| r / xi).collect()
    }
}

/// Generate the radial shells of `method` with an explicit radial scale $\xi$
/// (Bohr) and cutoff (Bohr).
///
/// A shell is active when $r \le \text{cutoff}$.  With `generate_all = false` the
/// generation stops after the first inactive shell for an ascending node
/// order (the named-grid construction); with `generate_all = true` the full
/// nominal set is returned and the tail is marked inactive.  Descending node
/// orders (PySCF `becke`) are always generated in full.
///
/// ```
/// use ruaa_dft_grids::radial::{Radial, shells};
///
/// let sh = shells(Radial::OrcaOptM3, 35, 0.8, 6.0, false);
/// assert!(!sh.is_empty());
/// assert!(sh.iter().any(|s| s.active));
/// assert!(sh.iter().all(|s| s.r >= 0.0));
/// ```
#[must_use]
pub fn shells(
    method: Radial,
    n_rad: usize,
    xi: f64,
    cutoff: f64,
    generate_all: bool,
) -> Vec<RadialShell> {
    let (rads, wrads) = method.grid_xi(n_rad, xi);
    let node_ratios = ratios(method, &rads, xi);
    let ascending = rads.windows(2).all(|pair| pair[0] <= pair[1]);
    let mut out = Vec::with_capacity(n_rad);
    for i in 0..n_rad {
        let active = rads[i] <= cutoff;
        out.push(RadialShell {
            r: rads[i],
            w: wrads[i],
            active,
            ratio: node_ratios[i],
        });
        if !active && !generate_all && ascending {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Radial; 8] = [
        Radial::TreutlerAhlrichs,
        Radial::Delley,
        Radial::MuraKnowles,
        Radial::GaussChebyshev,
        Radial::Becke,
        Radial::OrcaOptM3,
        Radial::OrcaM3,
        Radial::EulerMaclaurin,
    ];

    fn test_xi(method: Radial) -> f64 {
        match method.scale_policy() {
            RadialScalePolicy::Explicit => 1.0,
            _ => method.xi_for_element(6).unwrap(),
        }
    }

    #[test]
    fn radial_spec_matches_radial_presets() {
        for method in ALL {
            if method.scale_policy() == RadialScalePolicy::Explicit {
                let spec = method.spec();
                let (r, w) = spec.grid_xi(20, 1.0);
                assert_eq!((r, w), method.grid_xi(20, 1.0), "{method:?}");
                continue;
            }
            let preset = method.grid_for_element(20, 6).unwrap();
            let spec = method.spec().grid_for_element(20, 6).unwrap();
            assert_eq!(preset, spec, "{method:?}");
        }
    }

    #[test]
    fn explicit_scale_policy_needs_an_explicit_xi() {
        let spec = RadialSpec::new(Radial::TreutlerAhlrichs, RadialScalePolicy::Explicit);
        assert!(spec.grid_for_element(20, 6).is_err());
        let (r, w) = spec.grid_xi(20, 1.1);
        assert_eq!(r.len(), 20);
        assert_eq!(w.len(), 20);
    }

    struct LinearMap;

    impl RadialMap for LinearMap {
        fn radial_rd(&self, n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
            let dr = xi / n as f64;
            let r = (1..=n).map(|i| i as f64 * dr).collect();
            (r, vec![dr; n])
        }

        fn name(&self) -> &'static str {
            "linear"
        }
    }

    #[test]
    fn custom_map_uses_the_shared_spec() {
        let spec = RadialSpec::new(LinearMap, RadialScalePolicy::Explicit);
        let (r, w) = spec.grid_xi(4, 1.0);
        assert_eq!(spec.name(), "linear");
        assert_eq!(r, vec![0.25, 0.5, 0.75, 1.0]);
        assert!(w.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn methods_are_finite_and_monotone_for_light_atoms() {
        for method in ALL {
            for &z in &[1u32, 6, 18] {
                let xi = if method.scale_policy() == RadialScalePolicy::Explicit {
                    1.0
                } else {
                    method.xi_for_element(z).unwrap()
                };
                let (r, w) = method.grid_xi(20, xi);
                assert_eq!(r.len(), 20, "{method:?} z={z}");
                assert!(r.iter().all(|v| v.is_finite() && *v >= 0.0));
                assert!(w.iter().all(|v| v.is_finite()));
                // `becke` returns descending r; every other construction is
                // ascending.
                if method != Radial::Becke {
                    assert!(r[0] < r[19], "{method:?} z={z}: {:?}", &r[..2]);
                }
            }
        }
    }

    /// Every variant uses the single canonical convention:
    /// `grid_xi = 4*pi*r^2*dr` built from the raw `radial_rd` kernel.
    #[test]
    fn every_variant_uses_the_4pi_r2_dr_convention() {
        for method in ALL {
            let xi = test_xi(method);
            let (r, w) = method.grid_xi(20, xi);
            let (r2, dr) = method.radial_rd(20, xi);
            assert_eq!(r, r2, "{method:?}: node sets differ");
            for ((&ri, &wi), &dri) in r.iter().zip(&w).zip(&dr) {
                let want = 4.0 * crate::PI * ri * ri * dri;
                assert!(
                    (wi - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "{method:?}: {wi} vs 4*pi*r^2*dr = {want}"
                );
            }
        }
    }

    #[test]
    fn grid_is_grid_xi_with_the_element_preset() {
        for method in ALL {
            let z = 6u32;
            let xi = test_xi(method);
            if method.scale_policy() == RadialScalePolicy::Explicit {
                assert!(method.grid_for_element(20, z).is_err());
                continue;
            }
            let (r_z, w_z) = method.grid_for_element(20, z).unwrap();
            let (r_xi, w_xi) = method.grid_xi(20, xi);
            assert_eq!(r_z, r_xi, "{method:?}");
            assert_eq!(w_z, w_xi, "{method:?}");
        }
    }

    #[test]
    fn pyscf_accepts_an_explicit_xi() {
        // PySCF's Treutler table gives C (Z = 6) xi = 1.1.
        let (r_xi, w_xi) = Radial::TreutlerAhlrichs.grid_xi(20, 1.1);
        let (r_z, w_z) = Radial::TreutlerAhlrichs.grid_for_element(20, 6).unwrap();
        assert_eq!(r_xi, r_z);
        assert_eq!(w_xi, w_z);
    }

    #[test]
    fn radial_scale_selects_element_or_explicit_xi() {
        let method = Radial::TreutlerAhlrichs;
        let by_z = method.grid(20, RadialScale::Z(6)).unwrap();
        let by_element = method.grid_for_element(20, 6).unwrap();
        let by_xi = method.grid(20, RadialScale::Xi(1.1)).unwrap();
        let by_grid_xi = method.grid_xi(20, 1.1);
        assert_eq!(by_z, by_element);
        assert_eq!(by_z, by_xi);
        assert_eq!(by_xi, by_grid_xi);
        assert_eq!(RadialScale::Z(6).resolve(method), Ok(1.1));
        assert_eq!(RadialScale::Xi(2.0).resolve(method), Ok(2.0));

        // A map with no element rule accepts only an explicit scale.
        let by_xi = Radial::OrcaOptM3.grid(35, RadialScale::Xi(0.8)).unwrap();
        assert_eq!(by_xi, Radial::OrcaOptM3.grid_xi(35, 0.8));
        assert!(Radial::OrcaOptM3.grid(35, RadialScale::Z(1)).is_err());
    }

    #[test]
    fn grid_rejects_zero_shells_and_nonpositive_scale() {
        let method = Radial::TreutlerAhlrichs;
        assert!(matches!(
            method.grid_for_element(0, 6),
            Err(Error::InvalidArgument {
                argument: "radial shell count"
            })
        ));
        for xi in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert!(
                matches!(
                    method.grid(20, RadialScale::Xi(xi)),
                    Err(Error::InvalidArgument {
                        argument: "radial scale"
                    })
                ),
                "{xi}"
            );
        }
        assert!(method.grid_for_element(20, 6).is_ok());

        let spec = method.spec();
        assert!(matches!(
            spec.grid(0, RadialScale::Xi(6.0)),
            Err(Error::InvalidArgument {
                argument: "radial shell count"
            })
        ));
        assert!(matches!(
            spec.grid(20, RadialScale::Xi(f64::NAN)),
            Err(Error::InvalidArgument {
                argument: "radial scale"
            })
        ));
    }

    #[test]
    fn gc_variants_accept_an_explicit_xi() {
        let by_xi = Radial::OrcaOptM3.grid(35, RadialScale::Xi(0.8)).unwrap();
        assert_eq!(by_xi, Radial::OrcaOptM3.grid_xi(35, 0.8));
        assert!(by_xi.1.iter().all(|w| w.is_finite() && *w >= 0.0));
    }

    #[test]
    fn orca_radial_rd_is_finite_at_zero_scale() {
        for method in [Radial::OrcaOptM3, Radial::OrcaM3] {
            let (r, dr) = method.radial_rd(10, 0.0);
            assert!(r.iter().chain(&dr).all(|v| v.is_finite()), "{method:?}");
        }
    }

    #[test]
    fn gauss_chebyshev_matches_pyscf_probe() {
        // python: radi.gauss_chebyshev(5, 6)
        let (r, w) = Radial::GaussChebyshev.grid_for_element(5, 6).unwrap();
        let expect_r = [0.0085, 0.1953, 1.0, 2.9818, 7.4143];
        for (got, want) in r.iter().zip(expect_r) {
            assert!((got - want).abs() < 1e-3, "{got} vs {want}");
        }
        // w_rad = 4*pi*r^2*dr (PySCF rad_weight)
        assert!(w[2] > 0.0);
    }

    #[test]
    fn xi_presets_match_pyscf_and_orca_tables() {
        assert_eq!(Radial::TreutlerAhlrichs.xi_for_element(6), Ok(1.1));
        assert_eq!(Radial::MuraKnowles.xi_for_element(3), Ok(7.0));
        assert_eq!(Radial::MuraKnowles.xi_for_element(6), Ok(5.2));
        assert_eq!(Radial::Delley.xi_for_element(6), Ok(12.0));
        assert_eq!(Radial::GaussChebyshev.xi_for_element(6), Ok(1.0));
        assert_eq!(
            Radial::Becke.xi_for_element(1).unwrap(),
            crate::params::BRAGG_BOHR[1]
        );
        assert_eq!(
            Radial::Becke.xi_for_element(6).unwrap(),
            crate::params::BRAGG_BOHR[6] * 0.5
        );
        assert_eq!(
            Radial::OrcaOptM3.xi_for_element(6),
            Err(Error::UnsupportedGrid {
                name: "radial scale".to_string(),
                reason: "this policy has no element rule; supply an explicit xi".to_string(),
            })
        );
        assert!(Radial::EulerMaclaurin.xi_for_element(8).is_err());
    }

    #[test]
    fn euler_maclaurin_nodes_match_reference_values() {
        // (500, 6) H probe: r_0 = 3.999998406e-6, r_499 = 2.499999004e5
        let scale = 0.999999601419;
        let s = shells(Radial::EulerMaclaurin, 500, scale, 14.62, true);
        assert_eq!(s.len(), 500);
        assert!((s[0].r - 3.999998406e-6).abs() < 1e-12);
        assert!((s[499].r - 2.499999004e5).abs() < 1.0);
        // kept: r <= 14.62 -> 397 active shells
        assert_eq!(s.iter().filter(|s| s.active).count(), 397);
        // the ratio is the exact integer ratio ((i+1)/(n-i))^2, kept
        // separately to preserve boundary ties
        for (i, shell) in s.iter().enumerate() {
            let q = (i + 1) as f64 / (500 - i) as f64;
            assert!((shell.ratio - q * q).abs() < 1e-15);
        }
    }

    #[test]
    fn named_shell_generation_stops_after_cutoff() {
        // O (99, 590) named: 78 generated, 77 active
        let named = shells(Radial::EulerMaclaurin, 99, 0.888171278546, 10.36, false);
        assert_eq!(named.len(), 78);
        assert_eq!(named.iter().filter(|s| s.active).count(), 77);
        let all = shells(Radial::EulerMaclaurin, 99, 0.888171278546, 10.36, true);
        assert_eq!(all.iter().filter(|s| s.active).count(), 77);
        assert_eq!(all.len(), 99);
    }

    #[test]
    fn euler_maclaurin_ratio_ties_are_exact() {
        // 12/24 = 0.5, so the ratio 0.25 is an exact tie of the H/He row.
        assert_eq!(kernels::euler_maclaurin_ratios(35)[11], 0.25);
        let s = shells(Radial::EulerMaclaurin, 35, 1.0, f64::INFINITY, true);
        assert_eq!(s[11].ratio, 0.25);
    }
}
