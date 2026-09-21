#pragma once

#include "qcx/error.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/molecule/molecule.hpp"

#include <excgrid/error.hpp>
#include <excgrid/geometry.hpp>
#include <string_view>

namespace qcx::grid {

/// Translates an excgrid error code into a qcx Error carrying caller context.
///
/// excgrid reports two codes and no message; qcx errors carry a code and a
/// message, so every consumer of the standalone library funnels its failures
/// through here. The mapping is total: kInvalidArgument maps to
/// kInvalidArgument and kUnsupported to kUnimplemented.
/// \param code The code the excgrid call returned.
/// \param context What was being done, prefixed to the message.
/// \returns The translated error.
/// \ingroup qcx-grid
qcx::Error TranslateExcgridError(excgrid::ErrorCode code, std::string_view context);

/// Translates a molecule into excgrid's geometry aggregate.
///
/// The consumer side of the standalone boundary: excgrid's public headers
/// never see a qcx type, so species and Bohr coordinates are converted here.
/// Atom order is Molecule::Atoms() order - the canonical order the molecule
/// module established at construction - so a grid point's owning atom index
/// refers to the same atom list the caller holds.
/// \param molecule The molecule to translate.
/// \returns The geometry, or kInvalidArgument when the coordinate tensor is
/// not {atomCount, 3}.
/// \ingroup qcx-grid
qcx::Result<excgrid::Geometry> ToExcgridGeometry(const qcx::molecule::Molecule& molecule);

/// Translates grid settings into the library's build parameters.
///
/// The grid and its geometric derivatives must come from one parameter set, so
/// this is the single mapping from the settings the engine was built with to the
/// parameters a derivative provider is built with. A second copy of it would let
/// the two disagree about which points exist, and a provider refuses a block its
/// own parameters would not have produced.
/// \param settings The grid settings.
/// \returns The parameters, field for field.
/// \ingroup qcx-grid
[[nodiscard]] excgrid::GridParams ToExcgridParams(const XcGridSettings& settings) noexcept;

} // namespace qcx::grid
