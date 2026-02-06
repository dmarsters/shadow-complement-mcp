"""Shadow Complement Integration MCP Tool for Lushy.

Jungian shadow as visual vocabulary
After multi-domain composition (colimit blending), this tool applies shadow
complement to acknowledge what the persona denies, creating psychological depth
through systematic visual opposition.

Architecture:
- Layer 1: Domain ologs define complement_operations (antipodes)
- Layer 2: MCP loads YAML, computes antipodes (deterministic)
- Layer 3: Linear interpolation via integration_level

Usage:
    from shadow_complement_mcp import integrate_shadow_complement
    
    result = integrate_shadow_complement(
        unified_parameters={"tincture": "gules", "visual_weight": 0.8},
        aesthetic_domain="heraldic_blazonry",
        integration_level=0.5
    )

Mathematical Formula:
    integrated = persona + (shadow - persona) × integration_level
    
    integration_level ∈ [0, 1]:
        0.0 = pure persona
        0.5 = balanced acknowledgment
        1.0 = pure shadow

References:
    - heraldic_complement_operations_spec.yaml (template)
    - goguen_pushout_olog_spec.yaml (theoretical foundations)
    - Jung, C.G. "Psychology and Alchemy" (shadow psychology)
    - Fauconnier, G. & Turner, M. "The Way We Think" (vital relations)
"""

__version__ = "1.0.0"
__author__ = "Dal Marsters"
__author_email__ = "dal@lushy.app"
__license__ = "MIT"

from shadow_complement_mcp.server import (
    integrate_shadow_complement,
    list_available_domains,
    get_complement_operations_schema,
    explain_shadow_complement,
    load_olog,
    get_complement_operations,
    compute_antipode,
    interpolate_parameter,
)

__all__ = [
    "integrate_shadow_complement",
    "list_available_domains",
    "get_complement_operations_schema",
    "explain_shadow_complement",
    "load_olog",
    "get_complement_operations",
    "compute_antipode",
    "interpolate_parameter",
]
