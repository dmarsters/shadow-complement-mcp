#!/usr/bin/env python3
"""Shadow Complement Integration MCP Server.

Jungian shadow as visual vocabulary - applies psychological depth through
systematic visual opposition after multi-domain composition.

Architecture:
- Layer 1: Domain ologs define complement_operations (antipodes)
- Layer 2: This MCP loads YAML, computes antipodes (deterministic)
- Layer 3: Linear interpolation via integration_level

Mathematical Formula:
    integrated = persona + (shadow - persona) × integration_level
    
    integration_level ∈ [0, 1]:
        0.0 = pure persona
        0.5 = balanced acknowledgment
        1.0 = pure shadow
"""

from fastmcp import FastMCP
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import yaml
import re
import json

# Initialize FastMCP server
mcp = FastMCP("shadow-complement-integration")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# TODO: Update these paths to point to your actual domain ologs
DOMAIN_OLOG_PATHS = {
    "heraldic_blazonry": "/Users/dalmarsters/Documents/heraldic-blazonry-mcp/heraldic_blazonry_mcp/blazon_olog.yaml",
    "jazz_improvisation": "/Users/dalmarsters/Documents/jazz-improvisation-mcp/jazz_improvisation_mcp/categorical_structure.yaml",
    "cocktail_aesthetics": "/Users/dalmarsters/Documents/cocktail-aesthetics-mcp/cocktail_complement_operations.yaml",
    "norman_rockwell": "/Users/dalmarsters/Documents/norman-rockwell-mcp/src/norman_rockwell_mcp/norman_rockwell_complement_operations.yaml",  # NEW
    # Add more domains as they implement complement_operations
}

# Cache for loaded ologs (avoid repeated file I/O)
_olog_cache: Dict[str, Dict] = {}

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def load_olog(domain_name: str) -> Dict:
    """Load and cache a domain olog.
    
    Args:
        domain_name: Name of the domain (must be in DOMAIN_OLOG_PATHS)
        
    Returns:
        Parsed YAML olog dictionary
        
    Raises:
        FileNotFoundError: If domain not configured or file doesn't exist
        ValueError: If olog missing complement_operations section
    """
    if domain_name in _olog_cache:
        return _olog_cache[domain_name]
    
    if domain_name not in DOMAIN_OLOG_PATHS:
        raise FileNotFoundError(
            f"Domain '{domain_name}' not configured. "
            f"Available domains: {list(DOMAIN_OLOG_PATHS.keys())}"
        )
    
    olog_path = Path(DOMAIN_OLOG_PATHS[domain_name])
    if not olog_path.exists():
        raise FileNotFoundError(f"Olog file not found: {olog_path}")
    
    with open(olog_path) as f:
        olog = yaml.safe_load(f)
    
    if "complement_operations" not in olog:
        raise ValueError(
            f"Domain '{domain_name}' olog missing 'complement_operations' section. "
            "Cannot compute shadow complements."
        )
    
    _olog_cache[domain_name] = olog
    return olog


def get_complement_operations(domain_name: str) -> Dict:
    """Get complement operations section from domain olog.
    
    Args:
        domain_name: Name of the domain
        
    Returns:
        Dictionary of complement operations
    """
    olog = load_olog(domain_name)
    return olog["complement_operations"]


def compute_antipode(
    parameter_name: str,
    parameter_value: Union[str, float, int],
    complement_ops: Dict
) -> Tuple[Union[str, float], str]:
    """Compute the shadow complement (antipode) of a parameter value.
    
    Args:
        parameter_name: Name of the parameter
        parameter_value: Current value (persona)
        complement_ops: Complement operations from domain olog
        
    Returns:
        Tuple of (antipode_value, computation_type)
        computation_type is either "categorical_mapped" or "continuous_inversion"
        
    Raises:
        ValueError: If parameter not defined in complement_ops
    """
    if parameter_name not in complement_ops:
        raise ValueError(
            f"Parameter '{parameter_name}' not defined in complement_operations"
        )
    
    param_spec = complement_ops[parameter_name]
    param_type = param_spec.get("type", "categorical")
    
    if param_type == "categorical":
        # Categorical: direct lookup in mapping
        mapping = param_spec.get("mapping", {})
        if str(parameter_value) not in mapping:
            raise ValueError(
                f"Value '{parameter_value}' not in categorical mapping for '{parameter_name}'"
            )
        antipode = mapping[str(parameter_value)]
        return (antipode, "categorical_mapped")
    
    elif param_type == "continuous":
        # Continuous: use operation (typically "1 - value")
        operation = param_spec.get("operation", "1 - value")
        value_range = param_spec.get("range", [0.0, 1.0])
        
        # Normalize to [0, 1]
        normalized = (parameter_value - value_range[0]) / (value_range[1] - value_range[0])
        
        # Apply operation (currently only supporting "1 - value")
        if operation == "1 - value":
            complemented = 1.0 - normalized
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        # Denormalize back to original range
        antipode = value_range[0] + complemented * (value_range[1] - value_range[0])
        return (antipode, "continuous_inversion")
    
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")


def interpolate_parameter(
    persona_value: Union[str, float, int],
    shadow_value: Union[str, float, int],
    integration_level: float,
    parameter_name: str,
    parameter_type: str
) -> Union[str, float, int]:
    """Interpolate between persona and shadow values.
    
    Args:
        persona_value: Original value
        shadow_value: Complement value
        integration_level: [0.0, 1.0] where 0=pure persona, 1=pure shadow
        parameter_name: Name of parameter (for logging)
        parameter_type: "categorical" or "continuous"
        
    Returns:
        Interpolated value
    """
    if parameter_type == "continuous":
        # Linear interpolation: persona + (shadow - persona) × integration_level
        return persona_value + (shadow_value - persona_value) * integration_level
    
    elif parameter_type == "categorical":
        # Categorical: threshold at 0.5
        if integration_level < 0.5:
            return persona_value
        elif integration_level > 0.5:
            return shadow_value
        else:
            # Exactly 0.5: ambiguous/tension
            return f"{persona_value}↔{shadow_value}"
    
    else:
        raise ValueError(f"Unknown parameter type: {parameter_type}")


# ==============================================================================
# STRATEGIC ANALYSIS - Pattern Matching
# ==============================================================================

STRATEGIC_PATTERNS = {
    "authority_distribution": {
        "centralized": {
            "pattern": r"(?:single|central|unified|top-down|hierarchical|executive|leadership team|CEO|central authority)",
            "threshold": 3,
            "confidence": 0.75,
            "value": 0.1
        },
        "distributed": {
            "pattern": r"(?:distributed|decentralized|autonomous|self-organizing|team-based|flat|networked|collaborative)",
            "threshold": 3,
            "confidence": 0.75,
            "value": 0.9
        },
        "intermediate": {
            "pattern": r"(?:matrix|hybrid|federated|delegated|regional|divisional)",
            "threshold": 2,
            "confidence": 0.65,
            "value": 0.5
        }
    },
    "temporal_orientation": {
        "urgent_reactive": {
            "pattern": r"(?:immediate|urgent|crisis|now|critical|emergency|reactive|firefighting)",
            "threshold": 3,
            "confidence": 0.8,
            "temporal_focus": "short"
        },
        "long_term_patient": {
            "pattern": r"(?:long[- ]term|sustainable|generational|legacy|patient|gradual|enduring)",
            "threshold": 3,
            "confidence": 0.8,
            "temporal_focus": "long"
        },
        "planning_heavy": {
            "pattern": r"(?:strategic planning|roadmap|multi-year|phased approach|staged|milestone)",
            "threshold": 2,
            "confidence": 0.7,
            "temporal_focus": "structured"
        },
        "adaptive_responsive": {
            "pattern": r"(?:adaptive|agile|responsive|iterative|flexible|dynamic)",
            "threshold": 3,
            "confidence": 0.75,
            "temporal_focus": "fluid"
        }
    },
    "risk_appetite": {
        "risk_averse": {
            "pattern": r"(?:cautious|conservative|prudent|safe|proven|tested|low-risk|minimize risk)",
            "threshold": 3,
            "confidence": 0.75,
            "value": 0.2
        },
        "risk_seeking": {
            "pattern": r"(?:bold|aggressive|pioneering|breakthrough|disruptive|experimental|high-risk high-reward)",
            "threshold": 3,
            "confidence": 0.75,
            "value": 0.8
        },
        "calculated": {
            "pattern": r"(?:calculated|measured|balanced risk|risk-aware|controlled|strategic risk)",
            "threshold": 2,
            "confidence": 0.7,
            "value": 0.5
        }
    },
    "complexity_management": {
        "comprehensive_detailed": {
            "pattern": r"(?:comprehensive|detailed|thorough|exhaustive|granular|multi-faceted)",
            "threshold": 3,
            "confidence": 0.75
        },
        "minimal_essential": {
            "pattern": r"(?:minimal|essential|core|streamlined|simplified|focused)",
            "threshold": 3,
            "confidence": 0.75
        },
        "layered_nuanced": {
            "pattern": r"(?:layered|nuanced|sophisticated|multidimensional|integrated)",
            "threshold": 2,
            "confidence": 0.7
        }
    },
    "change_velocity": {
        "transformational_rapid": {
            "pattern": r"(?:transformation|rapid change|accelerated|fast-paced|aggressive timeline)",
            "threshold": 2,
            "confidence": 0.75
        },
        "incremental_stable": {
            "pattern": r"(?:incremental|gradual|steady|stable|evolutionary|step-by-step)",
            "threshold": 3,
            "confidence": 0.75
        },
        "disruptive_bold": {
            "pattern": r"(?:disruptive|revolutionary|radical|reimagine|reinvent)",
            "threshold": 2,
            "confidence": 0.8
        }
    },
    "stakeholder_emphasis": {
        "internal": {
            "pattern": r"(?:internal|employee|team|organizational culture|staff|workforce)",
            "threshold": 3,
            "confidence": 0.7,
            "value": 0.2
        },
        "external": {
            "pattern": r"(?:customer|client|market|external|stakeholder|user|competitive)",
            "threshold": 3,
            "confidence": 0.7,
            "value": 0.8
        },
        "balanced": {
            "pattern": r"(?:balanced|holistic|all stakeholders|comprehensive|inclusive)",
            "threshold": 2,
            "confidence": 0.65,
            "value": 0.5
        }
    }
}


def detect_authority_distribution(text: str) -> tuple[str, float, list[str]]:
    """Detect centralized vs distributed authority patterns."""
    text_lower = text.lower()
    patterns = STRATEGIC_PATTERNS["authority_distribution"]
    
    for pattern_name, pattern_def in patterns.items():
        matches = re.findall(pattern_def["pattern"], text_lower)
        if len(matches) >= pattern_def["threshold"]:
            evidence = list(set(matches))[:5]
            return (pattern_name, pattern_def["confidence"], evidence)
    
    return ("unclear", 0.0, [])


def detect_temporal_orientation(text: str) -> tuple[str, float, list[str]]:
    """Detect temporal planning orientation."""
    text_lower = text.lower()
    patterns = STRATEGIC_PATTERNS["temporal_orientation"]
    
    best_match = ("unclear", 0.0, [])
    best_score = 0
    
    for pattern_name, pattern_def in patterns.items():
        matches = re.findall(pattern_def["pattern"], text_lower)
        if len(matches) >= pattern_def["threshold"]:
            score = len(matches) * pattern_def["confidence"]
            if score > best_score:
                evidence = list(set(matches))[:5]
                best_match = (pattern_name, pattern_def["confidence"], evidence)
                best_score = score
    
    return best_match


def detect_risk_appetite(text: str) -> tuple[str, float, list[str]]:
    """Detect risk tolerance patterns."""
    text_lower = text.lower()
    patterns = STRATEGIC_PATTERNS["risk_appetite"]
    
    for pattern_name, pattern_def in patterns.items():
        matches = re.findall(pattern_def["pattern"], text_lower)
        if len(matches) >= pattern_def["threshold"]:
            evidence = list(set(matches))[:5]
            return (pattern_name, pattern_def["confidence"], evidence)
    
    return ("unclear", 0.0, [])


def detect_complexity_management(text: str) -> tuple[str, float, list[str]]:
    """Detect approach to complexity."""
    text_lower = text.lower()
    patterns = STRATEGIC_PATTERNS["complexity_management"]
    
    for pattern_name, pattern_def in patterns.items():
        matches = re.findall(pattern_def["pattern"], text_lower)
        if len(matches) >= pattern_def["threshold"]:
            evidence = list(set(matches))[:5]
            return (pattern_name, pattern_def["confidence"], evidence)
    
    return ("unclear", 0.0, [])


def detect_change_velocity(text: str) -> tuple[str, float, list[str]]:
    """Detect pace and nature of change."""
    text_lower = text.lower()
    patterns = STRATEGIC_PATTERNS["change_velocity"]
    
    best_match = ("unclear", 0.0, [])
    best_score = 0
    
    for pattern_name, pattern_def in patterns.items():
        matches = re.findall(pattern_def["pattern"], text_lower)
        if len(matches) >= pattern_def["threshold"]:
            score = len(matches) * pattern_def["confidence"]
            if score > best_score:
                evidence = list(set(matches))[:5]
                best_match = (pattern_name, pattern_def["confidence"], evidence)
                best_score = score
    
    return best_match


def detect_stakeholder_emphasis(text: str) -> tuple[str, float, list[str]]:
    """Detect stakeholder focus patterns."""
    text_lower = text.lower()
    patterns = STRATEGIC_PATTERNS["stakeholder_emphasis"]
    
    for pattern_name, pattern_def in patterns.items():
        matches = re.findall(pattern_def["pattern"], text_lower)
        if len(matches) >= pattern_def["threshold"]:
            evidence = list(set(matches))[:5]
            return (pattern_name, pattern_def["confidence"], evidence)
    
    return ("unclear", 0.0, [])


def analyze_strategy_document(strategy_text: str, confidence_threshold: float = 0.6) -> dict:
    """
    Analyze strategy document through shadow complement structural lens.
    
    Zero LLM cost - pure deterministic pattern matching.
    
    Args:
        strategy_text: Full text of strategy document
        confidence_threshold: Minimum confidence to report finding
        
    Returns:
        Dictionary with findings format for tomographic integration
    """
    findings = []
    
    # Run all detectors
    detectors = {
        "authority_distribution": detect_authority_distribution,
        "temporal_orientation": detect_temporal_orientation,
        "risk_appetite": detect_risk_appetite,
        "complexity_management": detect_complexity_management,
        "change_velocity": detect_change_velocity,
        "stakeholder_emphasis": detect_stakeholder_emphasis
    }
    
    for dimension, detector in detectors.items():
        pattern, confidence, evidence = detector(strategy_text)
        
        if confidence >= confidence_threshold:
            findings.append({
                "dimension": dimension,
                "pattern": pattern,
                "confidence": confidence,
                "evidence": evidence,
                "categorical_family": "constraints"
            })
    
    return {
        "domain": "shadow_complement_integration",
        "findings": findings,
        "total_findings": len(findings),
        "methodology": "deterministic_pattern_matching",
        "llm_cost_tokens": 0
    }


# ==============================================================================
# MCP TOOLS
# ==============================================================================

@mcp.tool()
def integrate_shadow_complement(
    unified_parameters: Dict[str, Any],
    aesthetic_domain: str,
    integration_level: float = 0.5
) -> Dict[str, Any]:
    """Apply Jungian shadow complement to unified composition parameters.
    
    This is the main tool for shadow integration. It takes parameters from
    a multi-domain composition (colimit) and applies shadow complement to
    acknowledge what the persona denies, creating psychological depth through
    systematic visual opposition.
    
    Args:
        unified_parameters: Parameter dictionary from colimit composition
        aesthetic_domain: Which domain's complement_operations to use
        integration_level: How much shadow to integrate [0.0, 1.0]
            0.0 = pure persona (original aesthetic)
            0.5 = balanced acknowledgment
            1.0 = pure shadow (inverse aesthetic)
    
    Returns:
        Dictionary containing:
        - persona: Original parameters
        - shadow_complements: Computed antipodes for each parameter
        - integrated_parameters: Interpolated result
        - integration_level: Echo of input
        - metadata: Processing details
    
    Example:
        result = integrate_shadow_complement(
            unified_parameters={
                "tincture": "gules",
                "visual_weight": 0.85,
                "detail_density": 0.7
            },
            aesthetic_domain="heraldic_blazonry",
            integration_level=0.5
        )
        
        # result["integrated_parameters"] will contain:
        # - tincture: "argent" (categorical flip at 0.5)
        # - visual_weight: 0.5 (linear interpolation)
        # - detail_density: 0.5 (linear interpolation)
    """
    if not 0.0 <= integration_level <= 1.0:
        raise ValueError("integration_level must be in [0.0, 1.0]")
    
    # Load complement operations for this domain
    complement_ops = get_complement_operations(aesthetic_domain)
    
    # Compute shadow complements
    shadow_complements = {}
    integrated_parameters = {}
    skipped_parameters = []
    warnings = []
    
    for param_name, param_value in unified_parameters.items():
        try:
            # Compute antipode
            antipode, computation_type = compute_antipode(
                param_name, param_value, complement_ops
            )
            shadow_complements[param_name] = antipode
            
            # Interpolate
            param_spec = complement_ops[param_name]
            param_type = param_spec.get("type", "categorical")
            
            integrated_value = interpolate_parameter(
                param_value,
                antipode,
                integration_level,
                param_name,
                param_type
            )
            integrated_parameters[param_name] = integrated_value
            
        except (ValueError, KeyError) as e:
            # Parameter not in complement_operations or invalid
            skipped_parameters.append(param_name)
            warnings.append(f"{param_name}: {str(e)}")
            # Keep original value
            integrated_parameters[param_name] = param_value
    
    return {
        "persona": unified_parameters,
        "shadow_complements": shadow_complements,
        "integrated_parameters": integrated_parameters,
        "integration_level": integration_level,
        "metadata": {
            "aesthetic_domain": aesthetic_domain,
            "parameters_processed": len(shadow_complements),
            "parameters_skipped": len(skipped_parameters),
            "skipped_list": skipped_parameters,
            "warnings": warnings
        }
    }


@mcp.tool()
def list_available_domains() -> Dict[str, str]:
    """List all configured domains with their status.
    
    Returns:
        Dictionary mapping domain names to status ("configured" or error message)
    """
    status = {}
    for domain_name, olog_path in DOMAIN_OLOG_PATHS.items():
        path = Path(olog_path)
        if not path.exists():
            status[domain_name] = f"ERROR: File not found at {olog_path}"
        else:
            try:
                load_olog(domain_name)
                status[domain_name] = "configured"
            except Exception as e:
                status[domain_name] = f"ERROR: {str(e)}"
    
    return status


@mcp.tool()
def get_complement_operations_schema(aesthetic_domain: str) -> Dict[str, Any]:
    """Get the structure of complement_operations for a domain.
    
    Useful for understanding what parameters can be complemented and how.
    
    Args:
        aesthetic_domain: Domain name
        
    Returns:
        Dictionary showing available parameters and their complement specs
    """
    complement_ops = get_complement_operations(aesthetic_domain)
    
    # Format for readability
    schema = {}
    for param_name, param_spec in complement_ops.items():
        schema[param_name] = {
            "type": param_spec.get("type", "categorical"),
            "psychological_principle": param_spec.get("psychological_principle", "N/A")
        }
        
        if param_spec.get("type") == "categorical":
            schema[param_name]["mappings_count"] = len(param_spec.get("mapping", {}))
        elif param_spec.get("type") == "continuous":
            schema[param_name]["range"] = param_spec.get("range", [0.0, 1.0])
            schema[param_name]["operation"] = param_spec.get("operation", "1 - value")
    
    return schema


@mcp.tool()
def explain_shadow_complement(
    parameter_name: str,
    parameter_value: Union[str, float],
    aesthetic_domain: str
) -> Dict[str, Any]:
    """Explain why a specific complement was chosen for a parameter.
    
    Args:
        parameter_name: Name of the parameter
        parameter_value: Current value
        aesthetic_domain: Domain name
        
    Returns:
        Dictionary with antipode and psychological explanation
    """
    complement_ops = get_complement_operations(aesthetic_domain)
    
    if parameter_name not in complement_ops:
        return {
            "error": f"Parameter '{parameter_name}' not found in {aesthetic_domain}",
            "available_parameters": list(complement_ops.keys())
        }
    
    param_spec = complement_ops[parameter_name]
    antipode, computation_type = compute_antipode(
        parameter_name, parameter_value, complement_ops
    )
    
    return {
        "parameter": parameter_name,
        "persona_value": parameter_value,
        "shadow_value": antipode,
        "computation_type": computation_type,
        "psychological_principle": param_spec.get("psychological_principle", "N/A"),
        "parameter_type": param_spec.get("type", "categorical")
    }


@mcp.tool()
def analyze_strategy_document_tool(strategy_text: str) -> str:
    """
    Analyze strategy document through shadow complement structural lens.
    
    Projects strategy text through shadow complement vocabulary to detect structural
    patterns: authority distribution (centralized/distributed), temporal orientation
    (urgent/long-term/adaptive), risk appetite (averse/seeking/calculated),
    complexity management (comprehensive/minimal/layered), change velocity
    (transformational/incremental/disruptive), and stakeholder emphasis
    (internal/external/balanced).
    
    This is LAYER 2 deterministic analysis with ZERO LLM cost - pure pattern
    matching against shadow complement taxonomy.
    
    Args:
        strategy_text: Full text of strategy document
        
    Returns:
        JSON string with findings format:
        {
            "domain": "shadow_complement_integration",
            "findings": [
                {
                    "dimension": "authority_distribution",
                    "pattern": "centralized",
                    "confidence": 0.75,
                    "evidence": ["single", "central", "unified", ...],
                    "categorical_family": "constraints"
                },
                ...
            ],
            "total_findings": 4,
            "methodology": "deterministic_pattern_matching",
            "llm_cost_tokens": 0
        }
    
    Example:
        >>> result = analyze_strategy_document_tool(strategy_pdf_text)
        >>> findings = json.loads(result)["findings"]
        >>> # Returns findings like:
        >>> # {"dimension": "authority_distribution", "pattern": "distributed",
        >>> #  "confidence": 0.75, "evidence": ["autonomous", "self-organizing"],
        >>> #  "categorical_family": "constraints"}
    
    Cost: 0 tokens (deterministic pattern matching)
    """
    result = analyze_strategy_document(strategy_text)
    return json.dumps(result, indent=2)


# ==============================================================================
# PHASE 2.6: SHADOW COMPLEMENT MORPHOSPACE & RHYTHMIC PRESETS
# ==============================================================================
#
# The shadow complement domain operates in a 5D parameter space capturing
# the Jungian individuation axis: how psychological shadow manifests visually.
#
# Parameters:
#   integration_level    - persona (0) → shadow (1) primary axis
#   psychological_tension - resolved (0) → maximally opposed (1)
#   visibility_gradient   - concealed (0) → fully exposed (1)
#   archetypal_depth      - surface/literal (0) → deep/symbolic (1)
#   projection_intensity  - internalized (0) → externally projected (1)
#
# ==============================================================================

SHADOW_PARAMETER_NAMES = [
    "integration_level",
    "psychological_tension",
    "visibility_gradient",
    "archetypal_depth",
    "projection_intensity"
]

# Canonical states in the shadow morphospace
SHADOW_CANONICAL_STATES = {
    "pure_persona": {
        "integration_level": 0.05,
        "psychological_tension": 0.10,
        "visibility_gradient": 0.05,
        "archetypal_depth": 0.10,
        "projection_intensity": 0.15,
        "description": "Fully conscious presentation - bright, ordered, controlled"
    },
    "glimpsed_shadow": {
        "integration_level": 0.25,
        "psychological_tension": 0.55,
        "visibility_gradient": 0.30,
        "archetypal_depth": 0.35,
        "projection_intensity": 0.40,
        "description": "First awareness of the unconscious - uncanny edges, slight distortion"
    },
    "active_confrontation": {
        "integration_level": 0.50,
        "psychological_tension": 0.95,
        "visibility_gradient": 0.70,
        "archetypal_depth": 0.65,
        "projection_intensity": 0.80,
        "description": "Direct encounter with shadow - maximum tension, stark duality"
    },
    "shadow_dominant": {
        "integration_level": 0.85,
        "psychological_tension": 0.60,
        "visibility_gradient": 0.90,
        "archetypal_depth": 0.80,
        "projection_intensity": 0.70,
        "description": "Shadow overtakes persona - dark, chthonic, raw unconscious material"
    },
    "transcendent_integration": {
        "integration_level": 0.50,
        "psychological_tension": 0.15,
        "visibility_gradient": 0.85,
        "archetypal_depth": 0.95,
        "projection_intensity": 0.30,
        "description": "Self realized through shadow work - deep, resolved, luminous darkness"
    },
    "enantiodromia_peak": {
        "integration_level": 0.50,
        "psychological_tension": 1.00,
        "visibility_gradient": 0.50,
        "archetypal_depth": 0.75,
        "projection_intensity": 0.95,
        "description": "Moment of reversal where opposites flip - maximum instability"
    },
    "penumbral_threshold": {
        "integration_level": 0.40,
        "psychological_tension": 0.45,
        "visibility_gradient": 0.50,
        "archetypal_depth": 0.50,
        "projection_intensity": 0.50,
        "description": "Liminal boundary between conscious and unconscious - twilight zone"
    }
}

# Phase 2.6 Rhythmic Presets - psychological oscillation patterns
SHADOW_RHYTHMIC_PRESETS = {
    "individuation_cycle": {
        "state_a": "pure_persona",
        "state_b": "shadow_dominant",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 24,
        "description": "Core Jungian individuation: smooth oscillation between persona and shadow"
    },
    "enantiodromia_pulse": {
        "state_a": "pure_persona",
        "state_b": "enantiodromia_peak",
        "pattern": "square",
        "num_cycles": 4,
        "steps_per_cycle": 12,
        "description": "Sudden reversals where opposites flip - sharp Jungian enantiodromia"
    },
    "depth_sounding": {
        "state_a": "glimpsed_shadow",
        "state_b": "transcendent_integration",
        "pattern": "triangular",
        "num_cycles": 3,
        "steps_per_cycle": 20,
        "description": "Progressive descent into archetypal depth, linear return to surface"
    },
    "tension_resolution": {
        "state_a": "active_confrontation",
        "state_b": "transcendent_integration",
        "pattern": "sinusoidal",
        "num_cycles": 5,
        "steps_per_cycle": 16,
        "description": "Oscillation between maximum tension and resolved integration"
    },
    "projection_withdrawal": {
        "state_a": "penumbral_threshold",
        "state_b": "active_confrontation",
        "pattern": "sinusoidal",
        "num_cycles": 4,
        "steps_per_cycle": 18,
        "description": "Cycles of projecting shadow outward then withdrawing to self-recognition"
    }
}


def _generate_oscillation(num_steps: int, num_cycles: float, pattern: str):
    """Generate oscillation pattern returning values in [0, 1].
    
    Args:
        num_steps: Total number of steps
        num_cycles: Number of complete A→B→A cycles
        pattern: "sinusoidal", "triangular", or "square"
    
    Returns:
        List of float alpha values [0, 1]
    """
    import math
    result = []
    for i in range(num_steps):
        t = 2.0 * math.pi * num_cycles * i / num_steps
        if pattern == "sinusoidal":
            alpha = 0.5 * (1.0 + math.sin(t))
        elif pattern == "triangular":
            t_norm = (t / (2.0 * math.pi)) % 1.0
            alpha = 2.0 * t_norm if t_norm < 0.5 else 2.0 * (1.0 - t_norm)
        elif pattern == "square":
            t_norm = (t / (2.0 * math.pi)) % 1.0
            alpha = 0.0 if t_norm < 0.5 else 1.0
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        result.append(alpha)
    return result


def _generate_preset_trajectory(preset_name: str) -> List[Dict[str, float]]:
    """Generate full trajectory for a Phase 2.6 preset.
    
    Args:
        preset_name: Name of preset in SHADOW_RHYTHMIC_PRESETS
    
    Returns:
        List of state dicts, one per step
    """
    if preset_name not in SHADOW_RHYTHMIC_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(SHADOW_RHYTHMIC_PRESETS.keys())}"
        )
    
    config = SHADOW_RHYTHMIC_PRESETS[preset_name]
    state_a = SHADOW_CANONICAL_STATES[config["state_a"]]
    state_b = SHADOW_CANONICAL_STATES[config["state_b"]]
    
    total_steps = config["num_cycles"] * config["steps_per_cycle"]
    alphas = _generate_oscillation(total_steps, config["num_cycles"], config["pattern"])
    
    trajectory = []
    for alpha in alphas:
        state = {}
        for p in SHADOW_PARAMETER_NAMES:
            state[p] = state_a[p] + (state_b[p] - state_a[p]) * alpha
        trajectory.append(state)
    
    return trajectory


# ==============================================================================
# PHASE 2.7: ATTRACTOR VISUALIZATION - SHADOW VISUAL VOCABULARY
# ==============================================================================
#
# Maps shadow morphospace coordinates to image-generation-ready keywords.
# Uses nearest-neighbor matching against canonical visual types.
#
# Visual Type Taxonomy:
#   persona_luminous   - Bright, ordered, clean surfaces, warm symmetry
#   penumbral_liminal  - Half-light, transitional, uncanny edges
#   shadow_chthonic    - Deep darkness, organic textures, asymmetric forms
#   integrated_numinous - Rich contrast, resolved opposites, luminous depth
#   enantiodromic_split - Stark duality, sharp divisions, maximum opposition
#
# ==============================================================================

SHADOW_VISUAL_TYPES = {
    "persona_luminous": {
        "coords": {
            "integration_level": 0.05,
            "psychological_tension": 0.10,
            "visibility_gradient": 0.05,
            "archetypal_depth": 0.10,
            "projection_intensity": 0.15
        },
        "keywords": [
            "brightly lit smooth surfaces",
            "bilateral symmetry",
            "clean geometric order",
            "warm diffused lighting",
            "polished reflective materials",
            "crisp defined edges",
            "luminous white and gold tones"
        ],
        "optical_properties": {
            "finish": "polished specular",
            "light_direction": "frontal even illumination",
            "contrast_ratio": "low, soft"
        },
        "color_associations": [
            "warm whites", "golds", "clear blues",
            "rose pinks", "clean ivory"
        ]
    },
    "penumbral_liminal": {
        "coords": {
            "integration_level": 0.35,
            "psychological_tension": 0.50,
            "visibility_gradient": 0.45,
            "archetypal_depth": 0.40,
            "projection_intensity": 0.45
        },
        "keywords": [
            "half-light transitional zone",
            "soft penumbral gradients",
            "partially obscured forms",
            "uncanny slight distortion",
            "liminal threshold space",
            "mist-veiled contours",
            "twilight ambiguity"
        ],
        "optical_properties": {
            "finish": "matte with selective sheen",
            "light_direction": "oblique raking light",
            "contrast_ratio": "medium, directional"
        },
        "color_associations": [
            "muted grays", "twilight blues", "dusty mauves",
            "amber half-tones", "tarnished silver"
        ]
    },
    "shadow_chthonic": {
        "coords": {
            "integration_level": 0.85,
            "psychological_tension": 0.60,
            "visibility_gradient": 0.90,
            "archetypal_depth": 0.80,
            "projection_intensity": 0.70
        },
        "keywords": [
            "deep chthonic darkness",
            "organic asymmetric textures",
            "rough weathered surfaces",
            "subterranean depth",
            "raw elemental materiality",
            "dense layered shadows",
            "primordial formlessness"
        ],
        "optical_properties": {
            "finish": "rough absorptive matte",
            "light_direction": "low underlighting or absence",
            "contrast_ratio": "extreme, localized"
        },
        "color_associations": [
            "deep charcoals", "earth blacks", "dried blood reds",
            "verdigris greens", "bruised purples"
        ]
    },
    "integrated_numinous": {
        "coords": {
            "integration_level": 0.50,
            "psychological_tension": 0.15,
            "visibility_gradient": 0.85,
            "archetypal_depth": 0.95,
            "projection_intensity": 0.30
        },
        "keywords": [
            "luminous darkness with inner glow",
            "resolved chiaroscuro balance",
            "deep symbolic resonance",
            "cathedral-light volumetric rays",
            "rich tonal complexity",
            "sacred geometry emerging from shadow",
            "gold leaf visible through dark patina"
        ],
        "optical_properties": {
            "finish": "satin with depth luminescence",
            "light_direction": "volumetric from within",
            "contrast_ratio": "full range, harmonized"
        },
        "color_associations": [
            "deep gold", "luminous indigo", "burgundy depth",
            "ember orange through smoke", "moon silver"
        ]
    },
    "enantiodromic_split": {
        "coords": {
            "integration_level": 0.50,
            "psychological_tension": 1.00,
            "visibility_gradient": 0.50,
            "archetypal_depth": 0.75,
            "projection_intensity": 0.95
        },
        "keywords": [
            "stark binary division",
            "sharp light-dark boundary at midline",
            "mirrored opposition across central axis",
            "high-contrast duality",
            "tense unresolved polarity",
            "fracture line between opposites",
            "double exposure superimposition"
        ],
        "optical_properties": {
            "finish": "split: polished vs rough",
            "light_direction": "hard side-lighting creating bisection",
            "contrast_ratio": "maximum, binary"
        },
        "color_associations": [
            "pure white vs pure black", "hot red vs cold blue",
            "gold vs tarnished iron", "flame vs ice"
        ]
    }
}


def _euclidean_distance_5d(coords_a: Dict[str, float], coords_b: Dict[str, float]) -> float:
    """Compute Euclidean distance between two 5D coordinate dicts."""
    total = 0.0
    for p in SHADOW_PARAMETER_NAMES:
        diff = coords_a.get(p, 0.0) - coords_b.get(p, 0.0)
        total += diff * diff
    return total ** 0.5


def _extract_shadow_visual_vocabulary(
    state: Dict[str, float],
    strength: float = 1.0
) -> Dict[str, Any]:
    """Map shadow parameter coordinates to nearest visual type and keywords.
    
    Uses nearest-neighbor matching against SHADOW_VISUAL_TYPES.
    
    Args:
        state: Parameter coordinates dict with shadow parameter names
        strength: Keyword weight multiplier [0.0, 1.0]
    
    Returns:
        Dict with nearest_type, distance, keywords, optical_properties, colors
    """
    min_dist = float('inf')
    nearest_type = None
    
    for type_name, type_spec in SHADOW_VISUAL_TYPES.items():
        dist = _euclidean_distance_5d(state, type_spec["coords"])
        if dist < min_dist:
            min_dist = dist
            nearest_type = type_name
    
    spec = SHADOW_VISUAL_TYPES[nearest_type]
    
    # Apply strength weighting to keyword selection
    if strength >= 0.7:
        keywords = spec["keywords"]
    elif strength >= 0.4:
        keywords = spec["keywords"][:5]
    else:
        keywords = spec["keywords"][:3]
    
    return {
        "nearest_type": nearest_type,
        "distance": round(min_dist, 4),
        "keywords": keywords,
        "optical_properties": spec["optical_properties"],
        "color_associations": spec["color_associations"],
        "strength": strength
    }


def _generate_shadow_prompt(
    state: Dict[str, float],
    style_modifier: str = "",
    strength: float = 1.0
) -> str:
    """Generate image-generation prompt from shadow parameter state.
    
    Args:
        state: Shadow parameter coordinates
        style_modifier: Optional prefix (e.g., "oil painting", "cinematic")
        strength: How strongly to weight shadow vocabulary
    
    Returns:
        Prompt string suitable for image generation
    """
    vocab = _extract_shadow_visual_vocabulary(state, strength)
    
    parts = []
    if style_modifier:
        parts.append(style_modifier)
    
    parts.extend(vocab["keywords"])
    
    # Add optical properties as descriptors
    optical = vocab["optical_properties"]
    parts.append(f"{optical['finish']} finish")
    parts.append(f"{optical['light_direction']}")
    parts.append(f"{optical['contrast_ratio']} contrast")
    
    # Add color palette
    if vocab["color_associations"]:
        color_str = ", ".join(vocab["color_associations"][:3])
        parts.append(f"color palette: {color_str}")
    
    return ", ".join(parts)


# ==============================================================================
# PHASE 2.6 MCP TOOLS
# ==============================================================================

@mcp.tool()
def get_shadow_morphospace_info() -> Dict[str, Any]:
    """Get complete shadow complement morphospace specification.
    
    Returns the 5D parameter space definition, all canonical states,
    available rhythmic presets, and visual vocabulary types.
    
    Cost: 0 tokens (pure taxonomy lookup)
    
    Returns:
        Dictionary with parameters, canonical_states, presets, visual_types
    """
    return {
        "domain": "shadow_complement",
        "parameter_names": SHADOW_PARAMETER_NAMES,
        "parameter_semantics": {
            "integration_level": "persona (0.0) → shadow (1.0), primary Jungian axis",
            "psychological_tension": "resolved (0.0) → maximally opposed (1.0)",
            "visibility_gradient": "concealed/repressed (0.0) → fully exposed (1.0)",
            "archetypal_depth": "surface/literal (0.0) → deep/symbolic (1.0)",
            "projection_intensity": "internalized (0.0) → externally projected (1.0)"
        },
        "canonical_states": {
            name: {
                "coords": {p: s[p] for p in SHADOW_PARAMETER_NAMES},
                "description": s["description"]
            }
            for name, s in SHADOW_CANONICAL_STATES.items()
        },
        "rhythmic_presets": {
            name: {
                "state_a": p["state_a"],
                "state_b": p["state_b"],
                "pattern": p["pattern"],
                "period": p["steps_per_cycle"],
                "total_steps": p["num_cycles"] * p["steps_per_cycle"],
                "description": p["description"]
            }
            for name, p in SHADOW_RHYTHMIC_PRESETS.items()
        },
        "visual_types": list(SHADOW_VISUAL_TYPES.keys()),
        "phase_2_6": True,
        "phase_2_7": True
    }


@mcp.tool()
def list_shadow_rhythmic_presets() -> Dict[str, Any]:
    """List all Phase 2.6 rhythmic presets for shadow complement domain.
    
    Each preset defines a psychological oscillation pattern between
    two canonical shadow states with specific waveform and timing.
    
    Cost: 0 tokens (pure taxonomy lookup)
    
    Returns:
        Dictionary of preset specifications
    """
    presets = {}
    for name, config in SHADOW_RHYTHMIC_PRESETS.items():
        presets[name] = {
            "state_a": config["state_a"],
            "state_b": config["state_b"],
            "pattern": config["pattern"],
            "period": config["steps_per_cycle"],
            "num_cycles": config["num_cycles"],
            "total_steps": config["num_cycles"] * config["steps_per_cycle"],
            "description": config["description"],
            "state_a_description": SHADOW_CANONICAL_STATES[config["state_a"]]["description"],
            "state_b_description": SHADOW_CANONICAL_STATES[config["state_b"]]["description"]
        }
    return {
        "domain": "shadow_complement",
        "presets": presets,
        "total_presets": len(presets),
        "available_patterns": ["sinusoidal", "triangular", "square"]
    }


@mcp.tool()
def generate_shadow_rhythmic_sequence(
    state_a_id: str,
    state_b_id: str,
    oscillation_pattern: str = "sinusoidal",
    num_cycles: int = 3,
    steps_per_cycle: int = 20,
    phase_offset: float = 0.0
) -> Dict[str, Any]:
    """Generate rhythmic oscillation between two shadow canonical states.
    
    Phase 2.6 temporal composition for shadow aesthetics. Creates periodic
    transitions cycling between shadow states, modeling psychological
    oscillation (e.g., individuation, projection-withdrawal, enantiodromia).
    
    Args:
        state_a_id: Starting canonical state name
        state_b_id: Alternating canonical state name
        oscillation_pattern: "sinusoidal", "triangular", or "square"
        num_cycles: Number of complete A→B→A cycles
        steps_per_cycle: Samples per cycle (= preset period)
        phase_offset: Starting phase [0.0, 1.0] where 0.0=state_a, 0.5=state_b
    
    Returns:
        Sequence with states, pattern info, and phase points
    
    Cost: 0 tokens (Layer 2 deterministic computation)
    """
    if state_a_id not in SHADOW_CANONICAL_STATES:
        raise ValueError(
            f"Unknown state '{state_a_id}'. "
            f"Available: {list(SHADOW_CANONICAL_STATES.keys())}"
        )
    if state_b_id not in SHADOW_CANONICAL_STATES:
        raise ValueError(
            f"Unknown state '{state_b_id}'. "
            f"Available: {list(SHADOW_CANONICAL_STATES.keys())}"
        )
    if oscillation_pattern not in ("sinusoidal", "triangular", "square"):
        raise ValueError(f"Unknown pattern: {oscillation_pattern}")
    
    state_a = SHADOW_CANONICAL_STATES[state_a_id]
    state_b = SHADOW_CANONICAL_STATES[state_b_id]
    
    total_steps = num_cycles * steps_per_cycle
    
    # Generate oscillation with phase offset
    import math
    sequence = []
    for i in range(total_steps):
        t = 2.0 * math.pi * num_cycles * i / total_steps
        t += 2.0 * math.pi * phase_offset  # Apply offset
        
        if oscillation_pattern == "sinusoidal":
            alpha = 0.5 * (1.0 + math.sin(t))
        elif oscillation_pattern == "triangular":
            t_norm = (t / (2.0 * math.pi)) % 1.0
            alpha = 2.0 * t_norm if t_norm < 0.5 else 2.0 * (1.0 - t_norm)
        else:  # square
            t_norm = (t / (2.0 * math.pi)) % 1.0
            alpha = 0.0 if t_norm < 0.5 else 1.0
        
        state = {}
        for p in SHADOW_PARAMETER_NAMES:
            state[p] = round(state_a[p] + (state_b[p] - state_a[p]) * alpha, 4)
        
        sequence.append({
            "step": i,
            "phase": round((i / total_steps) * num_cycles % 1.0, 4),
            "alpha": round(alpha, 4),
            "state": state
        })
    
    return {
        "domain": "shadow_complement",
        "state_a": state_a_id,
        "state_b": state_b_id,
        "oscillation_pattern": oscillation_pattern,
        "num_cycles": num_cycles,
        "steps_per_cycle": steps_per_cycle,
        "total_steps": total_steps,
        "phase_offset": phase_offset,
        "sequence": sequence,
        "state_a_description": state_a["description"],
        "state_b_description": state_b["description"]
    }


@mcp.tool()
def apply_shadow_rhythmic_preset(preset_name: str) -> Dict[str, Any]:
    """Apply a curated Phase 2.6 rhythmic preset.
    
    Pre-configured psychological oscillation patterns with optimal
    parameters for shadow complement aesthetics.
    
    Available presets:
        individuation_cycle:   persona ↔ shadow (smooth, period 24)
        enantiodromia_pulse:   persona ↔ enantiodromia peak (sharp, period 12)
        depth_sounding:        glimpsed ↔ integrated (triangular, period 20)
        tension_resolution:    confrontation ↔ integration (smooth, period 16)
        projection_withdrawal: threshold ↔ confrontation (smooth, period 18)
    
    Args:
        preset_name: Name of the rhythmic preset
    
    Returns:
        Complete rhythmic sequence with trajectory data
    
    Cost: 0 tokens (Layer 2 deterministic)
    """
    if preset_name not in SHADOW_RHYTHMIC_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(SHADOW_RHYTHMIC_PRESETS.keys())}"
        )
    
    config = SHADOW_RHYTHMIC_PRESETS[preset_name]
    
    # Generate trajectory
    trajectory = _generate_preset_trajectory(preset_name)
    
    return {
        "domain": "shadow_complement",
        "preset": preset_name,
        "description": config["description"],
        "state_a": config["state_a"],
        "state_b": config["state_b"],
        "pattern": config["pattern"],
        "period": config["steps_per_cycle"],
        "num_cycles": config["num_cycles"],
        "total_steps": len(trajectory),
        "trajectory": trajectory,
        "state_a_coords": {p: SHADOW_CANONICAL_STATES[config["state_a"]][p] for p in SHADOW_PARAMETER_NAMES},
        "state_b_coords": {p: SHADOW_CANONICAL_STATES[config["state_b"]][p] for p in SHADOW_PARAMETER_NAMES}
    }


# ==============================================================================
# PHASE 2.7 MCP TOOLS - ATTRACTOR VISUALIZATION
# ==============================================================================

@mcp.tool()
def extract_shadow_visual_vocabulary(
    state: Dict[str, float],
    strength: float = 1.0
) -> Dict[str, Any]:
    """Extract visual vocabulary from shadow parameter coordinates.
    
    Phase 2.7 tool: Maps a 5D shadow parameter state to the nearest
    canonical visual type and returns image-generation-ready keywords.
    
    Uses nearest-neighbor matching against 5 visual types derived from
    Jungian shadow psychology:
        persona_luminous     - bright, ordered, conscious presentation
        penumbral_liminal    - half-light, transitional, uncanny edges
        shadow_chthonic      - deep darkness, raw unconscious material
        integrated_numinous  - resolved opposites, luminous depth
        enantiodromic_split  - stark duality, maximum opposition
    
    Args:
        state: Parameter coordinates dict with keys:
            integration_level, psychological_tension, visibility_gradient,
            archetypal_depth, projection_intensity
        strength: Keyword weight multiplier [0.0, 1.0]
    
    Returns:
        Dict with nearest_type, distance, keywords, optical_properties,
        color_associations, strength
    
    Cost: 0 tokens (pure Layer 2 computation)
    
    Example:
        >>> extract_shadow_visual_vocabulary({
        ...     "integration_level": 0.50,
        ...     "psychological_tension": 0.95,
        ...     "visibility_gradient": 0.70,
        ...     "archetypal_depth": 0.65,
        ...     "projection_intensity": 0.80
        ... })
        {
            "nearest_type": "enantiodromic_split",
            "distance": 0.183,
            "keywords": ["stark binary division", "sharp light-dark boundary", ...],
            ...
        }
    """
    # Validate input
    for p in SHADOW_PARAMETER_NAMES:
        if p not in state:
            raise ValueError(
                f"Missing parameter '{p}'. Required: {SHADOW_PARAMETER_NAMES}"
            )
    
    strength = max(0.0, min(1.0, strength))
    return _extract_shadow_visual_vocabulary(state, strength)


@mcp.tool()
def generate_shadow_attractor_prompt(
    attractor_id: str = "",
    custom_state: Optional[Dict[str, float]] = None,
    mode: str = "composite",
    style_modifier: str = "",
    keyframe_count: int = 4
) -> Dict[str, Any]:
    """Generate image prompt from shadow attractor state or custom coordinates.
    
    Phase 2.7 tool: Translates shadow morphospace coordinates into visual
    prompts suitable for image generation (ComfyUI, Stable Diffusion, DALL-E).
    
    Modes:
        composite:  Single blended prompt from parameter state
        sequence:   Multiple keyframe prompts from a rhythmic preset trajectory
    
    Args:
        attractor_id: Preset attractor or rhythmic preset name.
            Attractor presets: "persona_luminous", "penumbral_liminal",
                "shadow_chthonic", "integrated_numinous", "enantiodromic_split"
            Rhythmic presets: "individuation_cycle", "enantiodromia_pulse",
                "depth_sounding", "tension_resolution", "projection_withdrawal"
            Use "" with custom_state for arbitrary coordinates.
        custom_state: Optional custom parameter coordinates dict.
            Overrides attractor_id if provided.
        mode: "composite" or "sequence"
        style_modifier: Optional prefix ("photorealistic", "oil painting", etc.)
        keyframe_count: Number of keyframes for sequence mode (default: 4)
    
    Returns:
        Dict with prompt(s), vocabulary details, and attractor metadata
    
    Cost: 0 tokens (Layer 2 deterministic)
    """
    if custom_state is not None:
        # Use custom coordinates directly
        for p in SHADOW_PARAMETER_NAMES:
            if p not in custom_state:
                raise ValueError(f"Missing parameter '{p}' in custom_state")
        
        state = custom_state
        source = "custom"
        source_name = "Custom State"
    
    elif attractor_id in SHADOW_VISUAL_TYPES:
        # Use a visual type's canonical coordinates
        state = SHADOW_VISUAL_TYPES[attractor_id]["coords"]
        source = "visual_type"
        source_name = attractor_id
    
    elif attractor_id in SHADOW_CANONICAL_STATES:
        # Use a canonical state's coordinates
        state = {p: SHADOW_CANONICAL_STATES[attractor_id][p] for p in SHADOW_PARAMETER_NAMES}
        source = "canonical_state"
        source_name = attractor_id
    
    elif attractor_id in SHADOW_RHYTHMIC_PRESETS and mode == "sequence":
        # Generate keyframes from rhythmic preset
        trajectory = _generate_preset_trajectory(attractor_id)
        total_steps = len(trajectory)
        
        keyframes = []
        for k in range(keyframe_count):
            step_idx = int(k * total_steps / keyframe_count) % total_steps
            kf_state = trajectory[step_idx]
            prompt = _generate_shadow_prompt(kf_state, style_modifier)
            vocab = _extract_shadow_visual_vocabulary(kf_state)
            
            keyframes.append({
                "keyframe": k,
                "step": step_idx,
                "prompt": prompt,
                "vocabulary": vocab,
                "state": kf_state
            })
        
        preset_config = SHADOW_RHYTHMIC_PRESETS[attractor_id]
        return {
            "mode": "sequence",
            "preset": attractor_id,
            "description": preset_config["description"],
            "period": preset_config["steps_per_cycle"],
            "total_steps": total_steps,
            "keyframe_count": keyframe_count,
            "style_modifier": style_modifier or "(none)",
            "keyframes": keyframes
        }
    
    elif attractor_id in SHADOW_RHYTHMIC_PRESETS and mode == "composite":
        # For composite mode on a preset, use midpoint state
        trajectory = _generate_preset_trajectory(attractor_id)
        mid_idx = len(trajectory) // 4  # Quarter-way = interesting transitional point
        state = trajectory[mid_idx]
        source = "rhythmic_preset_midpoint"
        source_name = attractor_id
    
    elif attractor_id == "":
        raise ValueError(
            "Provide either attractor_id or custom_state. "
            f"Available attractor IDs: {list(SHADOW_VISUAL_TYPES.keys())} (visual types), "
            f"{list(SHADOW_CANONICAL_STATES.keys())} (canonical states), "
            f"{list(SHADOW_RHYTHMIC_PRESETS.keys())} (rhythmic presets)"
        )
    else:
        raise ValueError(
            f"Unknown attractor_id '{attractor_id}'. "
            f"Available: {list(SHADOW_VISUAL_TYPES.keys())}, "
            f"{list(SHADOW_CANONICAL_STATES.keys())}, "
            f"{list(SHADOW_RHYTHMIC_PRESETS.keys())}"
        )
    
    # Composite mode: single prompt
    prompt = _generate_shadow_prompt(state, style_modifier)
    vocab = _extract_shadow_visual_vocabulary(state)
    
    return {
        "mode": "composite",
        "source": source,
        "source_name": source_name,
        "prompt": prompt,
        "vocabulary": vocab,
        "state": state,
        "style_modifier": style_modifier or "(none)"
    }


@mcp.tool()
def generate_shadow_sequence_prompts(
    preset_name: str,
    keyframe_count: int = 4,
    style_modifier: str = ""
) -> Dict[str, Any]:
    """Generate keyframe prompts from a Phase 2.6 shadow rhythmic preset.
    
    Phase 2.7 tool: Extracts evenly-spaced keyframes from a rhythmic
    oscillation sequence and generates an image prompt for each.
    
    Useful for:
        - Storyboard generation showing psychological arc
        - Animation keyframes for persona↔shadow transitions
        - Multi-panel visualization of individuation process
    
    Args:
        preset_name: Phase 2.6 preset name
        keyframe_count: Number of keyframes to extract (default: 4)
        style_modifier: Optional style prefix for all prompts
    
    Returns:
        Dict with keyframes, each containing step, state, prompt, vocabulary
    
    Cost: 0 tokens (Layer 2 deterministic)
    
    Example:
        >>> generate_shadow_sequence_prompts("individuation_cycle", keyframe_count=6)
        {
            "preset": "individuation_cycle",
            "keyframes": [
                {"step": 0, "prompt": "brightly lit smooth surfaces, ...", ...},
                {"step": 12, "prompt": "half-light transitional zone, ...", ...},
                {"step": 24, "prompt": "deep chthonic darkness, ...", ...},
                ...
            ]
        }
    """
    if preset_name not in SHADOW_RHYTHMIC_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(SHADOW_RHYTHMIC_PRESETS.keys())}"
        )
    
    trajectory = _generate_preset_trajectory(preset_name)
    total_steps = len(trajectory)
    config = SHADOW_RHYTHMIC_PRESETS[preset_name]
    
    keyframes = []
    for k in range(keyframe_count):
        step_idx = int(k * total_steps / keyframe_count) % total_steps
        kf_state = trajectory[step_idx]
        prompt = _generate_shadow_prompt(kf_state, style_modifier)
        vocab = _extract_shadow_visual_vocabulary(kf_state)
        
        keyframes.append({
            "keyframe": k,
            "step": step_idx,
            "prompt": prompt,
            "vocabulary": vocab,
            "state": kf_state
        })
    
    return {
        "domain": "shadow_complement",
        "preset": preset_name,
        "description": config["description"],
        "state_a": config["state_a"],
        "state_b": config["state_b"],
        "pattern": config["pattern"],
        "period": config["steps_per_cycle"],
        "total_steps": total_steps,
        "keyframe_count": keyframe_count,
        "style_modifier": style_modifier or "(none)",
        "keyframes": keyframes
    }


@mcp.tool()
def get_shadow_canonical_state(state_id: str) -> Dict[str, Any]:
    """Get complete specification for a shadow canonical state.
    
    Returns coordinates, description, nearest visual type, and keywords
    for a named canonical state.
    
    Args:
        state_id: Name of canonical state (e.g., "pure_persona", "shadow_dominant")
    
    Returns:
        Complete state specification with visual vocabulary
    
    Cost: 0 tokens (Layer 1 taxonomy lookup)
    """
    if state_id not in SHADOW_CANONICAL_STATES:
        raise ValueError(
            f"Unknown state '{state_id}'. "
            f"Available: {list(SHADOW_CANONICAL_STATES.keys())}"
        )
    
    state_spec = SHADOW_CANONICAL_STATES[state_id]
    coords = {p: state_spec[p] for p in SHADOW_PARAMETER_NAMES}
    vocab = _extract_shadow_visual_vocabulary(coords)
    
    return {
        "state_id": state_id,
        "coordinates": coords,
        "description": state_spec["description"],
        "visual_vocabulary": vocab,
        "prompt": _generate_shadow_prompt(coords)
    }


@mcp.tool()
def compute_shadow_distance(state_a_id: str, state_b_id: str) -> Dict[str, Any]:
    """Compute distance between two shadow canonical states.
    
    Layer 2: Pure distance computation (0 tokens).
    
    Args:
        state_a_id: First canonical state
        state_b_id: Second canonical state
    
    Returns:
        Distance value, per-parameter differences, and visual type shift
    """
    if state_a_id not in SHADOW_CANONICAL_STATES:
        raise ValueError(f"Unknown state '{state_a_id}'")
    if state_b_id not in SHADOW_CANONICAL_STATES:
        raise ValueError(f"Unknown state '{state_b_id}'")
    
    coords_a = {p: SHADOW_CANONICAL_STATES[state_a_id][p] for p in SHADOW_PARAMETER_NAMES}
    coords_b = {p: SHADOW_CANONICAL_STATES[state_b_id][p] for p in SHADOW_PARAMETER_NAMES}
    
    distance = _euclidean_distance_5d(coords_a, coords_b)
    
    per_param = {}
    for p in SHADOW_PARAMETER_NAMES:
        per_param[p] = round(coords_b[p] - coords_a[p], 4)
    
    vocab_a = _extract_shadow_visual_vocabulary(coords_a)
    vocab_b = _extract_shadow_visual_vocabulary(coords_b)
    
    return {
        "state_a": state_a_id,
        "state_b": state_b_id,
        "euclidean_distance": round(distance, 4),
        "per_parameter_difference": per_param,
        "visual_type_a": vocab_a["nearest_type"],
        "visual_type_b": vocab_b["nearest_type"],
        "visual_type_shift": vocab_a["nearest_type"] != vocab_b["nearest_type"]
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    mcp.run()
