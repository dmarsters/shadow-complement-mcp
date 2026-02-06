# Shadow Complement Integration MCP

Jungian shadow aesthetic integration for Lushy multi-domain compositions.

## Overview

This MCP server applies Jungian shadow complement to unified composition parameters, creating psychological depth through systematic visual opposition. After multi-domain composition (colimit blending), this tool acknowledges what the persona denies.

### Three-Layer Architecture

```
Layer 1: Categorical Structure
  ↓ YAML Olog with complement_operations section
  
Layer 2: Deterministic Mapping
  ↓ MCP loads YAML, computes antipodes (zero LLM cost)
  
Layer 3: Integration
  ↓ Linear interpolation via integration_level
```

### Mathematical Formula

```
integrated = persona + (shadow - persona) × integration_level

integration_level ∈ [0, 1]:
  0.0 = pure persona (original aesthetic)
  0.5 = balanced acknowledgment
  1.0 = pure shadow (inverse aesthetic)
```

## Installation

### Option 1: Local Development

```bash
# Clone or extract the archive
cd shadow-complement-integration

# Install with pip
pip install -e .

# Or with uv
uv pip install -e .
```

### Option 2: Deploy to FastMCP Cloud

1. Package the server:
```bash
tar -czf shadow-complement-integration.tar.gz shadow-complement-integration/
```

2. Upload to FastMCP Cloud
3. Configure entrypoint: `src/shadow_complement_mcp/server.py:mcp`

### Option 3: Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "shadow-complement-integration": {
      "command": "python",
      "args": [
        "-m",
        "shadow_complement_mcp.server"
      ],
      "env": {
        "PYTHONPATH": "/path/to/shadow-complement-integration/src"
      }
    }
  }
}
```

## Configuration

### Step 1: Configure Domain Paths

Edit `src/shadow_complement_mcp/server.py` and update `DOMAIN_OLOG_PATHS`:

```python
DOMAIN_OLOG_PATHS = {
    "heraldic_blazonry": "/path/to/heraldic_blazonry_mcp/olog/heraldic_blazonry.yaml",
    "jazz_improvisation": "/path/to/jazz_improvisation_mcp/olog/jazz_improvisation.yaml",
    "cocktail_aesthetics": "/path/to/cocktail_aesthetics_mcp/olog/cocktail_aesthetics.yaml",
}
```

### Step 2: Add complement_operations to Domain Ologs

Each domain olog needs a `complement_operations` section. See `examples/heraldic_complement_operations_spec.yaml` for template.

Example:
```yaml
complement_operations:
  tincture:
    type: categorical
    mapping:
      gules: argent
      azure: or
      sable: argent
    psychological_principle: "Chromatic opposition reveals denied warmth/coolness"
  
  visual_weight:
    type: continuous
    range: [0.0, 1.0]
    operation: "1 - value"
    psychological_principle: "Dominance ↔ submission"
```

## Usage

### Basic Usage

```python
from shadow_complement_mcp import integrate_shadow_complement

# After multi-domain composition
unified_params = {
    "tincture": "gules",
    "visual_weight": 0.85,
    "detail_density": 0.7
}

result = integrate_shadow_complement(
    unified_parameters=unified_params,
    aesthetic_domain="heraldic_blazonry",
    integration_level=0.5
)

# Access results
print(result["persona"])              # Original parameters
print(result["shadow_complements"])   # Computed antipodes
print(result["integrated_parameters"]) # Interpolated result
```

### Integration Levels

```python
# Persona-dominant (subtle shadow hints)
result_subtle = integrate_shadow_complement(
    unified_parameters=params,
    aesthetic_domain="heraldic_blazonry",
    integration_level=0.2
)

# Balanced acknowledgment
result_balanced = integrate_shadow_complement(
    unified_parameters=params,
    aesthetic_domain="heraldic_blazonry",
    integration_level=0.5
)

# Shadow-dominant
result_shadow = integrate_shadow_complement(
    unified_parameters=params,
    aesthetic_domain="heraldic_blazonry",
    integration_level=0.8
)
```

### Utility Tools

```python
# List configured domains
domains = list_available_domains()
# Returns: {"heraldic_blazonry": "configured", ...}

# Get complement schema
schema = get_complement_operations_schema("heraldic_blazonry")
# Returns parameter types and operations

# Explain specific complement
explanation = explain_shadow_complement(
    parameter_name="tincture",
    parameter_value="gules",
    aesthetic_domain="heraldic_blazonry"
)
# Returns psychological principle and antipode reasoning
```

## Composition Pipeline

```
Domain A  ──┐
            ├─→ [Colimit Composition] ──→ Unified Parameters
Domain B  ──┤                                    ↓
Domain C  ──┘                          [Shadow Integration]
                                                 ↓
                                          Final Output
```

Shadow integration is **endpoint-only** — it applies after all composition is complete, not within the functor chain.

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run test suite
pytest tests/

# Run specific test file
pytest tests/test_antipode_computation.py
```

## Examples

See `docs/EXAMPLE_USAGE.md` for complete examples including:
- Heraldic shadow integration
- Jazz improvisation shadow
- Multi-domain colimit + shadow
- Temporal shadow (animation sequences)

## Architecture Details

### Why Endpoint-Only?

Shadow integration applies only at final output, not during composition:
- Preserves blend's emergent properties
- No naturality squares to verify
- Clean separation: composition logic vs. integration logic

### Why Linear Interpolation?

Simple, reversible, intuitive:
- `integration_level=0` → pure persona
- `integration_level=0.5` → balanced
- `integration_level=1.0` → pure shadow

### Domain Requirements

Each domain must have:
1. Complete olog with categorical structure
2. `complement_operations` section with:
   - All parameters defined
   - Antipode mappings (categorical or continuous)
   - Psychological principles documented

## Contributing

To add a new domain:

1. Create `complement_operations` in domain olog
2. Add domain to `DOMAIN_OLOG_PATHS`
3. Run test suite to validate
4. Document psychological principles

See `docs/EXTENSION_GUIDE.md` for details.

## References

- Jung, C.G. "Psychology and Alchemy" (shadow psychology)
- Fauconnier, G. & Turner, M. "The Way We Think" (vital relations)
- Goguen, J. "Style as Choice of Blending Principles" (pushout formalism)

## License

MIT License - see LICENSE file

## Contact

- Author: Dal Marsters
- Email: dal@lushy.app
- Project: https://github.com/dmarsters/lushy
