# SPDX-License-Identifier: Apache-2.0
"""Muse Spark on gfx950: the kernels' shapes (``config``), the FlyDSL 0.3.2 bridge
(``compat``, installed on import), the plain-Python layouts (``layout``), the
stages (``stages``), the launches (``layer``) and the host runner (``runner``)."""

from sgl_amd_mono.muse_spark.gfx950 import compat  # noqa: F401  (installs the bridge first)
