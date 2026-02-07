# SLATE Schematic Diagram Generation SDK Specification

**Spec ID**: 011-schematic-diagram-sdk
**Status**: implementing
**Created**: 2026-02-07
**Author**: Claude Opus 4.5

## Overview

The Schematic Diagram Generation SDK provides a circuit-board style diagram generation framework for visualizing system architectures, data flows, and component relationships within the SLATE ecosystem. This SDK is part of SLATE's **Generative UI protocols** and produces GitHub-compatible SVG outputs using the locked theme (v3.0.0).

## Design Philosophy

### Schematic Principles
1. **Circuit-Board Aesthetic** - Technical precision with clean geometric lines
2. **Blueprint Engineering** - Dark backgrounds with light grid overlays
3. **Component-Based** - Modular, reusable node and connector types
4. **Layout Intelligence** - Automatic positioning with manual override support
5. **Theme Consistency** - Locked design tokens ensure visual coherence

### Integration Principles
1. **Zero External Dependencies** - All styles inline for GitHub compatibility
2. **Generative UI Protocol** - Dynamic generation based on system state
3. **Multi-Target Output** - Wiki, Pages, README, Dashboard support
4. **Real-Time Capability** - WebSocket-driven live updates

## Theme Specification (LOCKED)

### Core Identity
```
Name: SLATE Schematic Blueprint Theme
Version: 3.0.0 (LOCKED - inherits from design tokens)
Philosophy: Technical precision meets product elegance
```

### Color System (from design_tokens.py)
```css
/* Primary Brand */
--slate-primary: #B85A3C;         /* Anthropic-inspired warm rust */
--slate-primary-light: #D4785A;
--slate-primary-dark: #8B4530;

/* Blueprint Engineering */
--blueprint-bg: #0D1B2A;          /* Deep technical blue */
--blueprint-grid: #1B3A4B;        /* Subtle grid lines */
--blueprint-accent: #98C1D9;      /* Highlight color */
--blueprint-node: #E0FBFC;        /* Node backgrounds */

/* Status Semantics */
--status-active: #22C55E;         /* Green - connected/running */
--status-pending: #F59E0B;        /* Amber - in progress */
--status-error: #EF4444;          /* Red - failed/error */
--status-inactive: #6B7280;       /* Gray - stopped/offline */

/* Surface Colors */
--surface-dark: #1A1816;          /* Dark mode surface */
--surface-container: #2A2624;     /* Component backgrounds */
```

### Typography (Immutable)
```css
--font-display: 'Segoe UI', 'Inter', system-ui, sans-serif;
--font-mono: 'Consolas', 'JetBrains Mono', monospace;
```

## Component Library

### Node Components

| Component | Shape | Default Color | Use Case |
|-----------|-------|---------------|----------|
| `ServiceNode` | Rounded rect | Surface container | Microservices, servers |
| `DatabaseNode` | Cylinder | Info (#2196F3) | Data stores, caches |
| `GPUNode` | Hexagon | Warning (#FF9800) | GPU/compute resources |
| `AINode` | Rounded rect + brain | Primary (#B85A3C) | AI/ML services |
| `APINode` | Rectangle + port badge | Secondary | API endpoints |
| `QueueNode` | Parallelogram | Tertiary | Message queues |
| `ExternalNode` | Dashed border | Muted | External services |

### Connector Components

| Component | Style | Use Case |
|-----------|-------|----------|
| `FlowConnector` | Solid line + arrow | Direct data flow |
| `DashedConnector` | Dashed line + arrow | Optional/async flow |
| `BidirectionalConnector` | Line + double arrow | Two-way communication |
| `DataBus` | Thick line + taps | Multi-connection bus |

### Terminal Components

| Component | Shape | Use Case |
|-----------|-------|----------|
| `InputTerminal` | Circle + inward arrow | External input |
| `OutputTerminal` | Circle + outward arrow | External output |
| `PortTerminal` | Small circle | Connection port |

## Architecture

### Module Structure

```
slate/schematic_sdk/
├── __init__.py           # Public API exports
├── engine.py             # SchematicEngine - main orchestrator
├── components.py         # Component dataclasses
├── layout.py             # Layout algorithms
├── theme.py              # Theme integration
├── svg_renderer.py       # SVG generation pipeline
├── library.py            # Pre-built component definitions
├── exporters.py          # Export handlers
└── cli.py                # CLI entry point
```

### Core Classes

```python
@dataclass
class SchematicConfig:
    """Configuration for schematic generation."""
    width: int = 900
    height: int = 600
    title: str = "SLATE Schematic"
    theme: str = "blueprint"
    layout: str = "hierarchical"
    show_grid: bool = True
    show_legend: bool = True
    version_badge: str = ""

@dataclass
class Component:
    """Base component class."""
    id: str
    type: str
    label: str
    sublabel: str = ""
    position: Optional[Tuple[float, float]] = None
    status: str = "active"
    ports: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Connection:
    """Connection between components."""
    id: str
    from_node: str
    to_node: str
    from_port: str = "default"
    to_port: str = "default"
    label: str = ""
    style: str = "solid"
    animated: bool = False

class SchematicEngine:
    """Main schematic generation engine."""

    def __init__(self, config: SchematicConfig):
        self.config = config
        self.components: List[Component] = []
        self.connections: List[Connection] = []
        self.theme = ThemeManager(config.theme)
        self.layout_engine = LayoutEngine(config.layout)
        self.renderer = SVGRenderer(self.theme)

    def add_node(self, component: Component) -> None: ...
    def add_connector(self, connection: Connection) -> None: ...
    def apply_layout(self) -> None: ...
    def render_svg(self) -> str: ...
    def save(self, path: str, format: str = "svg") -> None: ...
```

## SVG Generation Pipeline

### Pipeline Stages

```
1. SCHEMA VALIDATION
   Input: Components + Connections
   Output: Validated component tree

2. LAYOUT CALCULATION
   Input: Component tree + Layout config
   Output: Positioned component tree (x, y resolved)

3. THEME APPLICATION
   Input: Positioned tree + Theme
   Output: Styled component tree (colors, fonts, strokes)

4. SVG GENERATION
   a. Generate <defs> (gradients, filters, patterns, markers)
   b. Render background + grid overlay
   c. Render connections (z-index: back)
   d. Render components (z-index: middle)
   e. Render labels/annotations (z-index: front)
   f. Render legend + version badge

5. POST-PROCESSING
   - Inline all styles (GitHub compatibility)
   - Add accessibility attributes
   - Optimize paths
```

### SVG Structure Template

```xml
<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"
     xmlns="http://www.w3.org/2000/svg"
     role="img" aria-label="{title}">

  <defs>
    <!-- Gradients -->
    <linearGradient id="bgGradient">...</linearGradient>
    <linearGradient id="primaryGradient">...</linearGradient>

    <!-- Filters -->
    <filter id="glow">...</filter>
    <filter id="shadow">...</filter>

    <!-- Markers -->
    <marker id="arrowhead">...</marker>

    <!-- Patterns -->
    <pattern id="grid">...</pattern>
  </defs>

  <!-- Background -->
  <rect class="background" fill="url(#bgGradient)"/>

  <!-- Grid Overlay -->
  <g class="grid-layer">...</g>

  <!-- Connections Layer -->
  <g class="connections-layer">...</g>

  <!-- Components Layer -->
  <g class="components-layer">...</g>

  <!-- Labels Layer -->
  <g class="labels-layer">...</g>

  <!-- Legend -->
  <g class="legend">...</g>

  <!-- Version Badge -->
  <g class="version-badge">...</g>
</svg>
```

## Layout Algorithms

### HierarchicalLayout

Best for: Service architectures, layer diagrams

```python
class HierarchicalLayout:
    """Arranges components in horizontal/vertical layers."""

    def __init__(
        self,
        direction: Literal["TB", "BT", "LR", "RL"] = "TB",
        layer_spacing: int = 120,
        node_spacing: int = 80,
        align: Literal["center", "left", "right"] = "center"
    ): ...

    def calculate_positions(
        self,
        nodes: List[Component],
        edges: List[Connection]
    ) -> Dict[str, Tuple[float, float]]:
        """
        1. Assign nodes to layers based on edge dependencies
        2. Minimize edge crossings within layers
        3. Center nodes within each layer
        4. Return position map
        """
```

### ForceDirectedLayout

Best for: Network diagrams, integration maps

```python
class ForceDirectedLayout:
    """Physics-based layout using spring forces."""

    def __init__(
        self,
        iterations: int = 100,
        repulsion_strength: float = 500,
        attraction_strength: float = 0.1,
        center_gravity: float = 0.1
    ): ...
```

### GridLayout

Best for: Component libraries, dashboards

```python
class GridLayout:
    """Snaps components to a grid."""

    def __init__(
        self,
        columns: int = 4,
        cell_width: int = 200,
        cell_height: int = 150,
        gap: int = 20
    ): ...
```

## Python SDK API

### Basic Usage

```python
from slate.schematic_sdk import (
    SchematicEngine, SchematicConfig,
    ServiceNode, AINode, GPUNode, DatabaseNode,
    FlowConnector, DataBus,
    HierarchicalLayout
)

# Create engine
config = SchematicConfig(
    width=900,
    height=600,
    title="SLATE System Architecture",
    theme="blueprint"
)
engine = SchematicEngine(config)

# Add components
engine.add_node(ServiceNode(
    id="dashboard",
    label="Dashboard",
    sublabel=":8080",
    status="active"
))

engine.add_node(AINode(
    id="ollama",
    label="Ollama",
    sublabel=":11434"
))

engine.add_node(GPUNode(
    id="gpu-cluster",
    label="Dual GPU",
    sublabel="RTX 5070 Ti x2"
))

# Add connections
engine.add_connector(FlowConnector(
    from_node="dashboard",
    to_node="ollama",
    label="API"
))

# Apply layout and render
engine.apply_layout()
svg = engine.render_svg()
engine.save("docs/assets/system-schematic.svg")
```

### Quick Generation Functions

```python
from slate.schematic_sdk import generate_system_diagram, generate_from_tech_tree

# One-liner for system diagram
svg = generate_system_diagram(
    title="SLATE Architecture",
    services=["Dashboard", "Ollama", "ChromaDB"],
    connections=[("Dashboard", "Ollama"), ("Ollama", "ChromaDB")]
)

# Generate from tech tree
svg = generate_from_tech_tree(".slate_tech_tree/tech_tree.json")
```

## CLI Tool

### Commands

```bash
# Generate from YAML definition
slate-schematic generate --input diagram.yaml --output diagram.svg

# Generate from current system state
slate-schematic from-system --output system.svg

# Generate from tech tree
slate-schematic from-tech-tree --output tech-tree.svg

# Batch generation
slate-schematic batch --dir specs/ --pattern "*/diagram.yaml"

# List available components
slate-schematic components --list

# Validate definition file
slate-schematic validate --input diagram.yaml
```

### YAML Definition Format

```yaml
diagram:
  title: "SLATE System Architecture"
  width: 900
  height: 600
  theme: blueprint
  layout: hierarchical

layers:
  - name: Presentation
    nodes:
      - type: service
        id: dashboard
        label: Dashboard
        sublabel: ":8080"
        status: active

  - name: AI Backends
    nodes:
      - type: ai
        id: ollama
        label: Ollama
        sublabel: ":11434"
      - type: gpu
        id: gpu-cluster
        label: Dual GPU

connections:
  - from: dashboard
    to: ollama
    type: flow
    label: API
```

## Integration Points

### MCP Tool Registration

```python
Tool(
    name="slate_schematic",
    description="Generate circuit-board style system diagrams",
    inputSchema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["generate", "from-system", "from-tech-tree"],
                "default": "from-system"
            },
            "output": {"type": "string"},
            "theme": {"type": "string", "enum": ["blueprint", "light", "dark"]},
            "layout": {"type": "string", "enum": ["hierarchical", "force", "grid"]}
        }
    }
)
```

### Dashboard API Endpoints

```python
@router.post("/api/schematic/generate")
async def generate_diagram(request: DiagramRequest) -> DiagramResponse:
    """Generate schematic from component definitions."""

@router.get("/api/schematic/from-system")
async def from_system() -> DiagramResponse:
    """Generate diagram from current system state."""

@router.get("/api/schematic/from-tech-tree")
async def from_tech_tree() -> DiagramResponse:
    """Generate diagram from tech tree."""

@router.websocket("/api/schematic/live")
async def live_updates(websocket: WebSocket):
    """Real-time diagram updates."""
```

### Generative UI Integration

```python
# In slate_generative_ui.py
class SchematicGenerator:
    """Generate schematics as part of Generative UI protocol."""

    async def generate_onboarding_schematic(
        self,
        step: OnboardingStep
    ) -> str:
        """Generate schematic showing current onboarding state."""

    async def generate_system_state_schematic(self) -> str:
        """Generate schematic from live system state."""
```

## Implementation Priority

### Phase 1: Core Engine
- [x] Specification document
- [ ] Module structure
- [ ] Base component classes
- [ ] SchematicEngine core
- [ ] Theme integration

### Phase 2: SVG Renderer
- [ ] Defs generation (gradients, filters)
- [ ] Component rendering
- [ ] Connection rendering
- [ ] Grid overlay
- [ ] Legend and badges

### Phase 3: Layout Algorithms
- [ ] HierarchicalLayout
- [ ] ForceDirectedLayout
- [ ] GridLayout

### Phase 4: Integration
- [ ] CLI tool
- [ ] MCP tool registration
- [ ] Dashboard API endpoints
- [ ] Claude Code slash command
- [ ] Generative UI protocol

### Phase 5: Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Visual verification
- [ ] Documentation

## Success Metrics

1. **GitHub Compatibility**: SVGs render correctly on GitHub without external dependencies
2. **Theme Consistency**: All diagrams match locked design tokens
3. **Generation Speed**: < 500ms for typical system diagrams
4. **Layout Quality**: No overlapping nodes, minimal edge crossings
5. **API Coverage**: All integration points functional

## Theme Lock Declaration

```
+---------------------------------------------------------------+
|                    THEME SPECIFICATION LOCK                    |
+---------------------------------------------------------------+
|                                                               |
|  Version: 3.0.0                                               |
|  Status: LOCKED (inherits from design_tokens.py)              |
|  Date: 2026-02-07                                             |
|                                                               |
|  The following are immutable:                                 |
|  - Blueprint color palette                                    |
|  - Status semantic colors                                     |
|  - Typography families                                        |
|  - Grid pattern dimensions                                    |
|                                                               |
|  Improvements must be additive, not breaking.                 |
|                                                               |
+---------------------------------------------------------------+
```
