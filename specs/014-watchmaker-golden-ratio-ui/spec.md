# Spec 014: Watchmaker Golden Ratio UI — Guided Operations Overhaul

# Modified: 2026-02-07T23:00:00Z | Author: COPILOT | Change: Initial spec creation

**Status**: implementing
**Priority**: P0 (Core System Ethos)
**Created**: 2026-02-07
**Author**: Copilot
**Depends On**: 007, 010, 012, 013

## Overview

This specification unifies SLATE's visual identity around two core metaphors:

1. **The Watchmaker** — Precision craftsmanship, visible mechanisms, layered depth
2. **The Golden Ratio (φ = 1.618...)** — Mathematical harmony governing all proportions

It also introduces a paradigm shift: **Guided Operations** — the interface distills
complex system management into buttons, choices, and visual feedback, reducing reliance
on typed prompts.

## Core Ethos

> **"Like a master horologist, every component has purpose. Like the golden ratio,
> every proportion has harmony. The system guides the operator — not the other way around."**

## The Watchmaker Principles (Exhaustive)

### 1. PRECISION — The Horologist's Eye
- Pixel-perfect alignment on 8px grid (ISO engineering module)
- No decorative elements without functional purpose
- Typography measured to mathematical ratios
- Component sizes derived from φ (golden ratio)

### 2. MECHANISM — The Visible Engine
- Users see the system working (rotating gears = active processes)
- Data flows rendered as animated traces (ISO 128 line types)
- Status shown as "jewels" (watch bearing stones = health indicators)
- Connection threads between components (spring-like connections)

### 3. DEPTH — The Layered Movement
- Information in discoverable layers (Surface → Mechanism → Components → Internals → Core)
- Z-depth perspective for 3D card effects
- Progressive disclosure: overview first, details on interaction

### 4. FUNCTION — Purpose-Driven Design
- Every UI element maps to a SLATE operation
- Buttons replace prompts for common operations
- Visual feedback confirms every action
- No dead-end screens

### 5. CRAFT — Beauty in Technical Precision
- The aesthetic emerges from functional perfection
- Copper/bronze tones reflect engineering materials
- Blueprint grid reflects engineering heritage
- Polish effects suggest machined surfaces

## The Golden Ratio System

### φ (Phi) = 1.6180339887...

Applied throughout the design:

```
┌─────────────────────────────────────────────────────────────────┐
│ GOLDEN RATIO PROPORTIONS                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Typography Scale (φ-derived):                                  │
│  ├── Display:   34px  (21 × φ)                                  │
│  ├── Headline:  21px  (13 × φ)                                  │
│  ├── Title:     16px  (base × φ⁰·³)                             │
│  ├── Body:      13px  (8 × φ)                                   │
│  ├── Label:     11px                                             │
│  └── Caption:    8px  (base unit)                                │
│                                                                 │
│  Spacing Scale (Fibonacci/φ):                                   │
│  1, 2, 3, 5, 8, 13, 21, 34, 55, 89 pixels                      │
│                                                                 │
│  Layout Ratios:                                                 │
│  ├── Hero split: 61.8% / 38.2%                                  │
│  ├── Sidebar: 38.2% width (of available space)                  │
│  ├── Content margins: Fibonacci progression                     │
│  └── Card aspect: 1.618:1                                       │
│                                                                 │
│  Animation Timing (φ-based):                                    │
│  ├── Flash:    100ms  (φ⁻⁴ seconds)                             │
│  ├── Quick:    200ms  (φ⁻³)                                     │
│  ├── Normal:   350ms  (φ⁻²)                                     │
│  ├── Smooth:   550ms  (φ⁻¹)                                     │
│  └── Dramatic: 900ms  (φ⁰)                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Unified Color Palette (Token Unification)

All 4 systems (Extension, VS Code Theme, GitHub Pages, Schematic SDK) MUST use
these exact values:

| Token | Value | Usage |
|-------|-------|-------|
| `primary` | `#B87333` | Brand copper, active elements |
| `primary-light` | `#C9956B` | Hover states, highlights |
| `primary-dark` | `#8B5E2B` | Pressed states, borders |
| `accent` | `#D4A054` | Gold accent, control flow |
| `success` | `#22C55E` | Active/healthy (jewel green) |
| `warning` | `#D4A054` | Pending/caution (jewel amber) |
| `error` | `#C47070` | Fault/error (jewel red) |
| `info` | `#7EA8BE` | Data flow (trace blue) |
| `surface` | `#0a0a0a` | Root background |
| `surface-variant` | `#111111` | Container background |
| `text-primary` | `#F5F0EB` | Primary text (warm white) |
| `text-secondary` | `#A8A29E` | Secondary text (warm gray) |
| `text-tertiary` | `#78716C` | Muted text (dim) |
| `copper-trace` | `#B87333` | Signal traces (ISO 128) |
| `data-trace` | `#7EA8BE` | Data flow traces |
| `power-trace` | `#C47070` | Power traces |
| `control-trace` | `#D4A054` | Control traces |
| `ground-trace` | `#78716C` | Ground/inactive traces |
| `blueprint-bg` | `#0D1B2A` | Blueprint sections |

## Guided Operations Paradigm

### Principle: Buttons > Prompts

The sidebar should function like a **mission control panel**:
- Every SLATE operation accessible via button/click
- Operations grouped into logical categories
- Visual feedback for running/completed operations
- Prompt input available but not required for common tasks

### Operation Groups

| Group | Operations | Visual |
|-------|-----------|--------|
| **Health** | Status, Runtime, Systems Check | Status jewels |
| **Services** | Start/Stop, Dashboard, Ollama | Toggle switches |
| **AI** | Models, Inference, Training | GPU meter |
| **CI/CD** | Dispatch, Monitor, Cancel | Workflow ring |
| **Workflow** | Status, Cleanup, Enforce | Task bar |
| **Security** | Scan, Audit, PII Check | Shield indicator |
| **Agents** | Registry, Health, Autonomous | Agent cards |

### Prompt Ingestion in Onboarding

New Step 8 in guided install:
```
8. PROMPT SETUP   → Configure @slate prompt preferences
   - Import custom system prompts
   - Set default @slate command mode
   - Configure guided vs. free-form preference
   - Link to AGENTS.md for agent routing
```

## Schematic Design Principles

### Engineering Drawing Standards (from Spec 013)
- ISO 128 line types for connections
- IEC 60617 symbols for components
- ASME Y14.44 reference designators
- PCB silkscreen labeling conventions

### Watchmaker Craft Details
- Gear SVG for spinning/processing indicators
- Jewel dots for status (inspired by watch jewel bearings)
- Mainspring coil for load indicators
- Balance wheel oscillation for activity pulses
- Escapement tick-tock for rate limiting
- Crown/winder icon for configuration access

## Application Surfaces

| Surface | Changes |
|---------|---------|
| `slateUnifiedDashboardView.ts` | New guided ops panel, φ tokens, prompt step |
| `slate-dark-color-theme.json` | Exhaustive token coverage, schematic scopes |
| `schematic_sdk/theme.py` | Unified palette, φ proportions |
| `docs/pages/index.html` | Token alignment (already close) |
| `package.json` | Version bump to 4.0.0 |

## Version

**Extension Version**: 4.0.0 (major overhaul)
**Theme Version**: 2.0.0

---

*This specification is a core SLATE ethos document. All visual systems must comply.*
