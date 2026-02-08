# Specification: Watchmaker Aesthetic & Interactive 3D Dashboard

**Spec ID**: 012-watchmaker-3d-dashboard
**Status**: implementing
**Created**: 2026-02-07
**Author**: Claude Opus 4.5
**Depends On**: 007-slate-design-system, 008-slate-guided-experience, 011-schematic-diagram-sdk

## Overview

This specification defines the **Watchmaker Design Philosophy** for SLATE's user interface, transforming the dashboard into a web-driven interactive 3D space that guides users through exhaustive, well-organized information structures using schematic-based organization.

The watchmaker metaphor represents:
- **Precision Engineering**: Every element has purpose and function
- **Intricate Mechanisms**: Visible interconnected systems
- **Craft & Artistry**: Technical beauty in functional design
- **Depth & Layers**: Multi-dimensional information architecture
- **Living Movement**: Animated gears, flows, and pulses

## Design Philosophy

### The Watchmaker Principles

```
                    ┌─────────────────────────────────────────┐
                    │        WATCHMAKER DESIGN TENETS          │
                    └─────────────────────────────────────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌─────────┐
   │PRECISION│   │MECHANISM │   │   DEPTH   │   │ FUNCTION │   │  CRAFT  │
   │         │   │          │   │           │   │          │   │         │
   │Pixel-   │   │Visible   │   │Layers of  │   │Every UI  │   │Beauty in│
   │perfect  │   │gears &   │   │detail you │   │element   │   │technical│
   │alignment│   │springs   │   │can zoom   │   │serves a  │   │precision│
   └─────────┘   └──────────┘   └───────────┘   └──────────┘   └─────────┘
```

### 1. PRECISION - The Horologist's Eye

**Every pixel serves a purpose. Nothing is arbitrary.**

```css
/* Precision Grid System */
--grid-unit: 4px;                    /* Base atomic unit */
--grid-micro: calc(var(--grid-unit) * 2);   /* 8px - micro spacing */
--grid-small: calc(var(--grid-unit) * 4);   /* 16px - component padding */
--grid-medium: calc(var(--grid-unit) * 8);  /* 32px - section spacing */
--grid-large: calc(var(--grid-unit) * 16);  /* 64px - major divisions */

/* Precision Alignment */
--align-tolerance: 0px;              /* No tolerance for misalignment */
--snap-to-grid: true;                /* All elements snap to 4px grid */
```

### 2. MECHANISM - The Visible Engine

**Users should see the system working, like gazing into an open watch case.**

Components that reveal mechanism:
- **Animated Gears**: Background elements showing system activity
- **Flow Lines**: Data paths visible as animated pulses
- **Status Indicators**: Beating heart of the system
- **Connection Threads**: Visible links between components
- **Rotating Elements**: Subtle rotation on active processes

```
     ╭───────────╮      ╭───────────╮      ╭───────────╮
     │ ◉ OLLAMA  │──────│ ◉ CHROMADB│──────│ ◉ DASHBOARD│
     │  ⟳ 11434  │      │  ⟳ 8000   │      │  ⟳ 8080   │
     ╰───────────╯      ╰───────────╯      ╰───────────╯
           │                  │                  │
           └────────┬─────────┴──────────┬───────┘
                    │    ◉ GPU CLUSTER   │
                    │  ⟳⟳ RTX 5070 Ti x2 │
                    └────────────────────┘
```

### 3. DEPTH - The Layered Watch Movement

**Information exists in discoverable layers, like examining a watch's complications.**

```
LAYER 0: Surface         │ Dashboard overview, key metrics
         ↓               │
LAYER 1: Mechanism       │ Service connections, data flows
         ↓               │
LAYER 2: Components      │ Individual service details
         ↓               │
LAYER 3: Internals       │ Configuration, logs, debugging
         ↓               │
LAYER 4: Core            │ System fundamentals, raw data
```

**3D Depth Implementation:**
```css
/* Z-Depth Layers (CSS perspective) */
--z-background: -200px;      /* Background decorative elements */
--z-grid: -100px;            /* Blueprint grid */
--z-connections: -50px;      /* Connection lines */
--z-components: 0px;         /* Primary UI components */
--z-floating: 50px;          /* Tooltips, modals */
--z-overlay: 100px;          /* Overlays, notifications */

/* Perspective container */
.dashboard-3d {
  perspective: 1200px;
  perspective-origin: 50% 50%;
  transform-style: preserve-3d;
}

/* Component depth positioning */
.component-layer {
  transform: translateZ(var(--z-components));
}

.connection-layer {
  transform: translateZ(var(--z-connections));
  opacity: 0.8;
}
```

### 4. FUNCTION - Purpose-Driven Design

**Like a watch complication, every element has a specific function.**

| Element | Function | Visual Representation |
|---------|----------|----------------------|
| Gear Icon | Processing indicator | Rotating when active |
| Pulse Line | Data flow | Animated stroke-dasharray |
| Status Jewel | Health indicator | Colored gem/dot |
| Balance Wheel | Activity oscillator | Pendulum motion |
| Mainspring | Power/load indicator | Coil visualization |
| Escapement | Rate limiter | Tick-tock animation |

### 5. CRAFT - Beauty in Precision

**The aesthetic emerges from functional perfection, not decoration.**

```css
/* Craft-quality details */
--border-precision: 1px solid var(--outline);     /* Exact borders */
--shadow-subtle: 0 1px 2px rgba(0,0,0,0.1);       /* Minimal shadows */
--glow-active: 0 0 8px var(--status-active);      /* Active glow */
--polish-reflection: linear-gradient(             /* Surface polish */
  135deg,
  rgba(255,255,255,0.15) 0%,
  rgba(255,255,255,0) 50%
);
```

## 3D Interactive Space Architecture

### Dashboard Structure

```
╔══════════════════════════════════════════════════════════════════════╗
║                        3D PERSPECTIVE CONTAINER                       ║
╠══════════════════════════════════════════════════════════════════════╣
║ ┌──────────────────────────────────────────────────────────────────┐ ║
║ │                    NAVIGATION ORBIT RING                         │ ║
║ │  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐        │ ║
║ │  │HOME│  │TASK│  │ AI │  │GPU │  │RUN │  │LOG │  │SET │        │ ║
║ │  └────┘  └────┘  └────┘  └────┘  └────┘  └────┘  └────┘        │ ║
║ └──────────────────────────────────────────────────────────────────┘ ║
║                                                                       ║
║     Z=-200: BACKGROUND GEARS (decorative, subtle rotation)           ║
║                     ⚙️ ⚙️ ⚙️ ⚙️ ⚙️ ⚙️                                    ║
║                                                                       ║
║     Z=-100: BLUEPRINT GRID                                            ║
║         ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐                        ║
║         ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤                        ║
║         └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘                        ║
║                                                                       ║
║     Z=-50: CONNECTION LAYER (animated data flows)                     ║
║              ════════════════════════════════                         ║
║             /          |          |          \                        ║
║                                                                       ║
║     Z=0: COMPONENT LAYER (main UI elements)                           ║
║     ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐  ║
║     │   ◉ SERVICES    │   │   ◉ AI MODELS   │   │   ◉ GPU FARM    │  ║
║     │   ═════════════ │   │   ═════════════ │   │   ═════════════ │  ║
║     │   Dashboard  ✓  │   │   mistral-nemo  │   │   RTX 5070 Ti   │  ║
║     │   Ollama     ✓  │   │   llama3.2      │   │   RTX 5070 Ti   │  ║
║     │   ChromaDB   ✓  │   │   phi           │   │   Load: 45%     │  ║
║     └─────────────────┘   └─────────────────┘   └─────────────────┘  ║
║                                                                       ║
║     Z=50: FLOATING LAYER (tooltips, status badges)                    ║
║                      ┌─────────────┐                                  ║
║                      │ Status: OK  │                                  ║
║                      └─────────────┘                                  ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Schematic-Based Organization

All information structures follow schematic/blueprint organization:

```
                    ┌─────────────────────────────────────┐
                    │       SCHEMATIC ORGANIZATION         │
                    └─────────────────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
    ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
    │   HIERARCHY   │       │    CIRCUIT    │       │   DATAFLOW    │
    │               │       │               │       │               │
    │ Top → Down    │       │ Nodes + Lines │       │ Input→Output  │
    │ Tree view     │       │ Connection    │       │ Pipelines     │
    │ Nested boxes  │       │ topology      │       │ Arrows        │
    └───────────────┘       └───────────────┘       └───────────────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     ▼
                    ┌─────────────────────────────────────┐
                    │     COMBINED: SCHEMATIC DASHBOARD    │
                    └─────────────────────────────────────┘
```

### Information Architecture Tiers

```yaml
TIER 1 - SYSTEM OVERVIEW:
  location: Dashboard Home
  depth: Surface
  content:
    - System health ring (animated)
    - Key metrics (4 cards)
    - Quick actions (floating)
    - Connection status map

TIER 2 - SERVICE CONSTELLATION:
  location: Services View
  depth: Mechanism
  content:
    - Service nodes (schematic layout)
    - Inter-service connections (animated lines)
    - Health status (jewel indicators)
    - Port/endpoint labels

TIER 3 - AI WORKBENCH:
  location: AI Models View
  depth: Mechanism
  content:
    - Model cards (specs, status)
    - Inference queue visualization
    - GPU allocation diagram
    - Token usage meters

TIER 4 - TASK ORCHESTRATION:
  location: Workflows View
  depth: Components
  content:
    - Task queue (Kanban schematic)
    - Workflow pipelines (animated)
    - Execution timeline
    - Status progression

TIER 5 - CONFIGURATION CORE:
  location: Settings
  depth: Internals
  content:
    - Configuration tree
    - Environment variables
    - Security settings
    - Integration toggles
```

## Visual Components

### Gear Component (Watchmaker Signature)

```svg
<!-- Animated Gear SVG -->
<svg class="gear" viewBox="0 0 100 100">
  <defs>
    <linearGradient id="gearGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#B85A3C"/>
      <stop offset="100%" style="stop-color:#8B4530"/>
    </linearGradient>
  </defs>
  <g class="gear-body" style="transform-origin: 50% 50%">
    <!-- 8 teeth -->
    <path d="M50,5 L55,15 L45,15 Z" fill="url(#gearGrad)"/>
    <!-- ... 7 more teeth at 45° intervals -->
    <!-- Central circle -->
    <circle cx="50" cy="50" r="25" fill="url(#gearGrad)"/>
    <circle cx="50" cy="50" r="10" fill="#0D1B2A"/>
  </g>
  <style>
    .gear-body { animation: spin 10s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</svg>
```

### Pulse Line (Data Flow)

```css
.data-flow-line {
  stroke: var(--status-active);
  stroke-width: 2;
  stroke-dasharray: 10 5;
  animation: pulse-flow 1.5s linear infinite;
}

@keyframes pulse-flow {
  to { stroke-dashoffset: -15; }
}
```

### Status Jewel (Health Indicator)

```css
.status-jewel {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: radial-gradient(
    circle at 30% 30%,
    var(--status-active) 0%,
    color-mix(in srgb, var(--status-active) 60%, black) 100%
  );
  box-shadow:
    0 0 4px var(--status-active),
    inset 0 1px 2px rgba(255,255,255,0.3);
}

.status-jewel.error {
  --status-active: var(--status-error);
}

.status-jewel.pending {
  --status-active: var(--status-pending);
  animation: jewel-pulse 1.5s ease-in-out infinite;
}

@keyframes jewel-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.7; transform: scale(1.1); }
}
```

### 3D Card Component

```css
.watchmaker-card {
  /* Base styling */
  background: var(--surface-container);
  border: 1px solid var(--outline-variant);
  border-radius: 16px;
  padding: var(--grid-medium);

  /* 3D positioning */
  transform: translateZ(0);
  transform-style: preserve-3d;
  transition: transform 0.3s var(--easing-spring);

  /* Polish effect */
  background-image: var(--polish-reflection);

  /* Subtle shadow for depth */
  box-shadow:
    0 4px 8px rgba(0,0,0,0.15),
    0 1px 2px rgba(0,0,0,0.1);
}

.watchmaker-card:hover {
  transform: translateZ(20px) rotateX(-2deg);
  box-shadow:
    0 12px 24px rgba(0,0,0,0.2),
    0 4px 8px rgba(0,0,0,0.15);
}

/* Card header with gear accent */
.watchmaker-card-header {
  display: flex;
  align-items: center;
  gap: var(--grid-small);
  margin-bottom: var(--grid-small);
}

.watchmaker-card-header .gear-icon {
  width: 24px;
  height: 24px;
  opacity: 0.6;
}

.watchmaker-card.active .gear-icon {
  animation: spin 8s linear infinite;
}
```

## Dashboard Sections

### 1. Command Center (Home)

```
╔══════════════════════════════════════════════════════════════════════╗
║  ⚙️ S.L.A.T.E. COMMAND CENTER                    [◉ All Systems OK]  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║   ┌─────────────────────────────────────────────────────────────┐    ║
║   │                  SYSTEM HEALTH RING                          │    ║
║   │                                                              │    ║
║   │         ╭──────╮    ╭──────╮    ╭──────╮    ╭──────╮       │    ║
║   │     ───( GPU  )────( AI   )────(TASKS )────(RUNNER)───      │    ║
║   │         ╰──────╯    ╰──────╯    ╰──────╯    ╰──────╯       │    ║
║   │                        │                                     │    ║
║   │                   ╭────┴────╮                                │    ║
║   │                   │  SLATE  │                                │    ║
║   │                   │   ⚙️    │                                │    ║
║   │                   ╰─────────╯                                │    ║
║   └─────────────────────────────────────────────────────────────┘    ║
║                                                                       ║
║   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ ║
║   │ ◉ Services   │  │ ◉ AI Models  │  │ ◉ Task Queue │  │ ◉ Runner │ ║
║   │              │  │              │  │              │  │          │ ║
║   │  7/7 Active  │  │  10 Models   │  │  3 Pending   │  │ 4 Online │ ║
║   │  ▓▓▓▓▓▓▓▓▓▓  │  │  ▓▓▓▓▓▓░░░░  │  │  ▓▓▓░░░░░░░  │  │ ▓▓▓▓░░░░ │ ║
║   └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘ ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 2. Service Constellation

```
╔══════════════════════════════════════════════════════════════════════╗
║  ⚙️ SERVICE CONSTELLATION                          [7 Services]       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║                      ┌─────────────────┐                             ║
║                      │   ◉ Dashboard   │                             ║
║                      │     :8080       │                             ║
║                      └────────┬────────┘                             ║
║                               │                                       ║
║              ┌────────────────┼────────────────┐                     ║
║              │                │                │                      ║
║      ┌───────┴───────┐  ┌────┴────┐   ┌──────┴──────┐               ║
║      │   ◉ Ollama    │  │◉ Chroma │   │ ◉ MCP Server│               ║
║      │    :11434     │──│  :8000  │───│    :stdio   │               ║
║      └───────────────┘  └─────────┘   └─────────────┘               ║
║              │                │                                       ║
║      ┌───────┴───────┐  ┌────┴─────────┐                             ║
║      │  ◉ Foundry    │  │ ◉ GPU Farm   │                             ║
║      │    :5272      │  │  2x RTX 5070 │                             ║
║      └───────────────┘  └──────────────┘                             ║
║                                                                       ║
║  ════════════════════════════════════════════════════════════════    ║
║  Legend: ◉ Active  ◎ Standby  ○ Offline  ─── Data Flow               ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 3. GPU Workbench

```
╔══════════════════════════════════════════════════════════════════════╗
║  ⚙️ GPU WORKBENCH                              [Dual RTX 5070 Ti]     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  ┌────────────────────────────┐  ┌────────────────────────────┐     ║
║  │  GPU 0: RTX 5070 Ti        │  │  GPU 1: RTX 5070 Ti        │     ║
║  │  ════════════════════════  │  │  ════════════════════════  │     ║
║  │                            │  │                            │     ║
║  │  ⟳ Compute: ▓▓▓▓▓▓░░ 75%  │  │  ⟳ Compute: ▓▓▓░░░░░ 38%  │     ║
║  │  ◐ Memory:  ▓▓▓▓░░░░ 50%  │  │  ◐ Memory:  ▓▓░░░░░░ 25%  │     ║
║  │  ◉ Power:   ▓▓▓▓▓░░░ 62%  │  │  ◉ Power:   ▓▓▓░░░░░ 40%  │     ║
║  │                            │  │                            │     ║
║  │  Active Tasks:             │  │  Active Tasks:             │     ║
║  │  ├─ mistral-nemo (7B)      │  │  ├─ llama3.2 (3B)          │     ║
║  │  └─ embedding gen          │  │  └─ (idle)                 │     ║
║  │                            │  │                            │     ║
║  │  Temp: 58°C  CUDA 12.8     │  │  Temp: 42°C  CUDA 12.8     │     ║
║  └────────────────────────────┘  └────────────────────────────┘     ║
║                                                                       ║
║  Load Distribution: GPU 0 ▓▓▓▓▓▓▓░░░ ← Scheduler → ░░░▓▓▓▓ GPU 1    ║
╚══════════════════════════════════════════════════════════════════════╝
```

## CSS Variables (Watchmaker Extension)

```css
:root {
  /* === WATCHMAKER EXTENSION === */

  /* Gear Colors */
  --gear-primary: var(--slate-primary);
  --gear-secondary: var(--slate-primary-dark);
  --gear-accent: var(--blueprint-accent);

  /* Mechanism Animation */
  --gear-speed-slow: 20s;
  --gear-speed-medium: 10s;
  --gear-speed-fast: 5s;

  /* 3D Perspective */
  --perspective-distance: 1200px;
  --z-background: -200px;
  --z-grid: -100px;
  --z-connections: -50px;
  --z-components: 0px;
  --z-floating: 50px;
  --z-overlay: 100px;

  /* Precision Grid */
  --grid-unit: 4px;
  --snap-grid: 4px;

  /* Jewel Status */
  --jewel-size: 12px;
  --jewel-glow-spread: 4px;

  /* Polish Effect */
  --polish-reflection: linear-gradient(
    135deg,
    rgba(255,255,255,0.15) 0%,
    rgba(255,255,255,0) 50%
  );

  /* Precision Borders */
  --border-hairline: 0.5px;
  --border-thin: 1px;
  --border-medium: 2px;

  /* Watchmaker Shadows */
  --shadow-inset: inset 0 1px 2px rgba(0,0,0,0.1);
  --shadow-pressed: inset 0 2px 4px rgba(0,0,0,0.15);
  --shadow-gear: 0 2px 4px rgba(0,0,0,0.2);
}
```

## Implementation Requirements

### Dashboard Updates

1. **3D Container**: Wrap dashboard in perspective container
2. **Z-Layer System**: Implement depth layers for visual hierarchy
3. **Gear Animations**: Add rotating gear elements for activity indication
4. **Flow Lines**: Animated connection lines between components
5. **Jewel Indicators**: Status gems replacing simple dots
6. **Schematic Layout**: Blueprint-style organization
7. **Polish Effects**: Subtle reflections and shadows

### Required Files

```
slate/
├── static/
│   ├── css/
│   │   ├── watchmaker.css        # Watchmaker design tokens
│   │   └── 3d-dashboard.css      # 3D positioning styles
│   └── js/
│       ├── gear-animation.js     # Gear rotation controller
│       └── depth-manager.js      # Z-layer management
├── templates/
│   └── dashboard-3d.html         # Main 3D dashboard template
└── schematic_sdk/                # Already exists from spec 011
```

## Theme Lock Declaration

```
+---------------------------------------------------------------+
|              WATCHMAKER THEME SPECIFICATION LOCK               |
+---------------------------------------------------------------+
|                                                               |
|  Version: 4.0.0                                               |
|  Status: LOCKED (extends design_tokens.py)                    |
|  Date: 2026-02-07                                             |
|                                                               |
|  The following are immutable:                                 |
|  - Gear animation speeds                                      |
|  - Z-layer depth values                                       |
|  - Grid unit (4px)                                            |
|  - Jewel status colors                                        |
|  - Perspective distance                                       |
|                                                               |
|  Additive improvements only. No breaking changes.             |
|                                                               |
+---------------------------------------------------------------+
```

## Success Metrics

1. **Visual Coherence**: All elements follow watchmaker principles
2. **Functional Clarity**: Every element's purpose is immediately clear
3. **Depth Perception**: 3D layering enhances information hierarchy
4. **Performance**: Animations < 16ms frame time (60fps)
5. **Accessibility**: All interactive elements keyboard-navigable
6. **Schematic Consistency**: Organization matches blueprint style

---

## GitHub Pages Requirements (LOCKED)

### Page Ethos

The SLATE GitHub Pages site is the **public face** of the project. It must:
- Present **objective project health** information
- Show **verifiable metrics** from public sources (GitHub API)
- Demonstrate **professional engineering** quality
- Follow all SLATE security protocols

### Content Security Rules

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    PAGE SECURITY REQUIREMENTS                          ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  MUST NOT INCLUDE:                                                     ║
║  ────────────────────────────────────────────────────────────────────  ║
║  ✗ Local system information (GPU specs, CPU, RAM)                      ║
║  ✗ Personal user data or paths                                         ║
║  ✗ API keys, tokens, or credentials                                    ║
║  ✗ Internal IP addresses or ports (except documentation examples)      ║
║  ✗ Runtime status of local services                                    ║
║  ✗ User-specific configuration                                         ║
║  ✗ Debug or development information                                    ║
║                                                                        ║
║  MUST INCLUDE:                                                         ║
║  ────────────────────────────────────────────────────────────────────  ║
║  ✓ Project description and purpose                                     ║
║  ✓ Feature list (capabilities, not implementation)                     ║
║  ✓ Installation instructions (generic)                                 ║
║  ✓ Documentation links                                                 ║
║  ✓ GitHub project health metrics (public API data only)                ║
║  ✓ Roadmap and version information                                     ║
║  ✓ License and attribution                                             ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### Project Health Metrics (Objective Data)

Display only metrics verifiable from public GitHub data:

| Metric | Source | Display |
|--------|--------|---------|
| Repository Stars | GitHub API | Live counter badge |
| Fork Count | GitHub API | Badge |
| Open Issues | GitHub API | Count with trend |
| Open PRs | GitHub API | Count |
| Last Commit | GitHub API | Relative date |
| CI Status | GitHub Actions | Pass/Fail badge |
| License | Repository | Badge |
| Contributors | GitHub API | Count |

### Page Structure Requirements

```
PAGE ARCHITECTURE (LOCKED)
══════════════════════════════════════════════════════════════════════

1. HERO SECTION
   ├── Project name and tagline
   ├── Primary value proposition (LOCAL AI, FREE, YOUR HARDWARE)
   ├── CTA buttons (Get Started, Documentation)
   └── Watchmaker mechanism visual (decorative gears)

2. PROJECT HEALTH DASHBOARD
   ├── GitHub stats badges (stars, forks, issues)
   ├── Build status indicator
   ├── Latest release version
   └── Community activity pulse

3. FEATURE SHOWCASE
   ├── Core capabilities (not implementation details)
   ├── Schematic diagrams of architecture (conceptual)
   ├── Use case examples
   └── Integration highlights

4. ARCHITECTURE OVERVIEW
   ├── System diagram (conceptual, not live)
   ├── Component relationships
   ├── Data flow visualization
   └── Technology stack

5. REQUIREMENTS & INSTALLATION
   ├── Minimum requirements (generic)
   ├── Recommended setup (generic)
   ├── One-line install command
   └── Link to full documentation

6. ROADMAP
   ├── Current version features
   ├── Upcoming releases
   ├── Community contributions welcome
   └── Issue tracker link

7. FOOTER
   ├── License information
   ├── Attribution
   ├── Links (GitHub, Wiki, Issues)
   └── Project activity indicator (decorative)
```

### Visual Design Lock

```css
/* LOCKED PAGE DESIGN TOKENS */
:root {
  /* Brand Colors (locked) */
  --page-primary: #B85A3C;        /* Warm rust */
  --page-surface: #1A1816;        /* Dark earth */
  --page-surface-variant: #2A2624;
  --page-on-surface: #E8E2DE;

  /* Blueprint Accent (locked) */
  --page-blueprint-bg: #0D1B2A;
  --page-blueprint-accent: #98C1D9;

  /* Watchmaker Elements (locked) */
  --page-gear-opacity: 0.12;      /* Background gears */
  --page-gear-speed: 20s;         /* Slow rotation */
  --page-jewel-glow: 4px;         /* Status indicators */

  /* No local-data styling */
  --page-no-runtime-data: true;   /* Enforced by design */
}
```

### Accessibility Requirements

1. **WCAG 2.1 AA Compliance**
   - Contrast ratio 4.5:1 minimum for text
   - Focus indicators on all interactive elements
   - Skip navigation link
   - Semantic HTML structure

2. **Performance**
   - First Contentful Paint < 1.5s
   - No JavaScript required for core content
   - CSS animations use `transform` and `opacity` only
   - Prefers-reduced-motion support

<!-- Modified: 2026-02-09T06:55:00Z | Author: COPILOT | Change: Add static performance audit metrics -->
### Performance Audit (Static)

Static audit metrics for GitHub Pages assets:

- `docs/pages/index.html`: 60,185 bytes
- `docs/pages/css/watchmaker.css`: 12,447 bytes
- Inline blocks: 2 style, 2 script
- Animation mentions: 4 (index), 16 (watchmaker.css)
- Keyframes: 7 (watchmaker.css)

Note: 60fps verification requires manual browser profiling and remains pending.

3. **SEO**
   - Semantic `<h1>` through `<h6>` hierarchy
   - Meta description
   - Open Graph tags
   - Structured data (JSON-LD)

### Page Ethos Declaration

```
+═══════════════════════════════════════════════════════════════════════+
║                    SLATE PAGE ETHOS (LOCKED)                           ║
+═══════════════════════════════════════════════════════════════════════+
║                                                                        ║
║  1. OBJECTIVITY                                                        ║
║     The page presents verifiable, objective information only.          ║
║     No marketing hyperbole. No unsubstantiated claims.                 ║
║                                                                        ║
║  2. TRANSPARENCY                                                       ║
║     All metrics come from public, verifiable sources.                  ║
║     GitHub badges link to their data source.                           ║
║                                                                        ║
║  3. SECURITY                                                           ║
║     Zero local data. Zero runtime information.                         ║
║     No personal information. No internal paths.                        ║
║                                                                        ║
║  4. PROFESSIONALISM                                                    ║
║     Engineering-quality presentation.                                  ║
║     Watchmaker precision in every pixel.                               ║
║     No decorative elements without purpose.                            ║
║                                                                        ║
║  5. ACCESSIBILITY                                                      ║
║     Readable by all users. Keyboard navigable.                         ║
║     Screen reader compatible. Reduced motion support.                  ║
║                                                                        ║
+═══════════════════════════════════════════════════════════════════════+
```

<!-- Modified: 2026-02-09T06:35:00Z | Author: COPILOT | Change: Add Phase 7 documentation update notes -->
### Documentation Update (Phase 7)

This specification now includes the operational documentation for the watchmaker information architecture:

- **Breadcrumb + tier navigation**: Tracks current section and depth tier with surface-mechanism-components-internals-core labels.
- **Drill-down interactions**: Clickable `.drilldown-trigger` expands `.drilldown-detail` panels and updates breadcrumb context.
- **Zoom/focus interactions**: Double-click focus containers to isolate a panel; click outside to clear focus.
- **Schematic alignment**: Information architecture maps to schematic sections for consistent navigation and UI semantics.
- **Generative UI alignment**: Watchmaker surfaces are compatible with the schematic SDK protocol in Generative UI flows.
