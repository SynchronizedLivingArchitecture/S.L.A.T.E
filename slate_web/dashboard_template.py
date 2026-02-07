#!/usr/bin/env python3
# Modified: 2026-02-07T13:00:00Z | Author: COPILOT | Change: ProArt copper/B&W menu-driven dashboard redesign
# Purpose: Generates the SLATE dashboard HTML template with ProArt-inspired design
"""
SLATE Dashboard Template Builder
=================================
Generates the complete DASHBOARD_HTML string for slate_dashboard_server.py

Design principles:
  1. ASUS ProArt Design Language: Copper accents, true black substrate, precision aesthetics
  2. Anthropic geometric art: Crystalline tessellations, network constellations
  3. Menu-driven guided UX: Assume user is NOT a systems engineer or AI expert
  4. Two control layers: Dashboard (background CLI) + SLATE Controls (local Copilot)
  5. Data-dense editorial layout with warm-white typography on true-black surfaces

The template is self-contained (inline CSS/JS, no external deps) for 127.0.0.1 serving.
"""

from pathlib import Path
from slate_web.design_system import SlateDesignTokens, SlateLogoGenerator, GeometricPatternGenerator

import base64


def _svg_to_data_uri(svg: str) -> str:
    """Convert SVG string to data URI for CSS background-image."""
    encoded = base64.b64encode(svg.encode('utf-8')).decode('ascii')
    return f"data:image/svg+xml;base64,{encoded}"


def build_template() -> str:
    """Build the complete dashboard HTML template."""

    # Generate assets inline
    logo_icon_svg = SlateLogoGenerator.generate(48, "icon")
    constellation_svg = GeometricPatternGenerator.constellation_grid(1920, 1080, 60, seed=42)
    constellation_uri = _svg_to_data_uri(constellation_svg)

    tokens = SlateDesignTokens.generate_tokens("dark")

    # Build CSS custom properties string
    token_css_lines = []
    for key, value in tokens.items():
        token_css_lines.append(f"            {key}: {value};")
    tokens_css = "\n".join(token_css_lines)

    return f'''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'self' http://127.0.0.1:* ws://127.0.0.1:*; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self' http://127.0.0.1:* ws://127.0.0.1:*; img-src 'self' data:;">
    <title>S.L.A.T.E.</title>
    <style>
        /* ═══════════════════════════════════════════════════════════════
           SLATE Design System v4.0
           ASUS ProArt + Anthropic Geometric + Menu-Driven UX
           ═══════════════════════════════════════════════════════════════ */

        :root {{
{tokens_css}
        }}

        /* ─── Reset & Base ─────────────────────────────────────────── */
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        html {{
            scroll-behavior: smooth;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            font-size: 15px;
        }}

        body {{
            font-family: var(--sl-font-sans);
            background: var(--sl-bg-root);
            color: var(--sl-text-primary);
            min-height: 100vh;
            line-height: 1.6;
            overflow-x: hidden;
        }}

        /* ─── Geometric Background ─────────────────────────────────── */
        body::before {{
            content: '';
            position: fixed;
            inset: 0;
            background:
                url("{constellation_uri}") center/cover no-repeat,
                radial-gradient(ellipse at 20% 0%, rgba(184,115,51,0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(184,115,51,0.03) 0%, transparent 50%);
            opacity: 0.6;
            pointer-events: none;
            z-index: 0;
        }}

        /* Scanline overlay for depth */
        body::after {{
            content: '';
            position: fixed;
            inset: 0;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0,0,0,0.015) 2px,
                rgba(0,0,0,0.015) 4px
            );
            pointer-events: none;
            z-index: 0;
        }}

        /* ─── Layout Shell ─────────────────────────────────────────── */
        .shell {{
            position: relative;
            z-index: 1;
            max-width: 1440px;
            margin: 0 auto;
            padding: var(--sl-space-6) var(--sl-space-8);
        }}

        /* ─── Topbar ───────────────────────────────────────────────── */
        .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: var(--sl-space-4) 0;
            margin-bottom: var(--sl-space-8);
            border-bottom: 1px solid var(--sl-border);
        }}

        .topbar-brand {{
            display: flex;
            align-items: center;
            gap: var(--sl-space-4);
        }}

        .topbar-logo {{
            width: 40px;
            height: 40px;
            flex-shrink: 0;
        }}

        .topbar-title {{
            display: flex;
            flex-direction: column;
        }}

        .topbar-title h1 {{
            font-size: var(--sl-space-5);
            font-weight: 700;
            letter-spacing: 0.12em;
            color: var(--sl-text-primary);
            line-height: 1.1;
        }}

        .topbar-title .subtitle {{
            font-size: 0.65rem;
            color: var(--sl-text-tertiary);
            letter-spacing: 0.2em;
            text-transform: uppercase;
        }}

        .topbar-actions {{
            display: flex;
            align-items: center;
            gap: var(--sl-space-3);
        }}

        .topbar-status {{
            display: flex;
            align-items: center;
            gap: var(--sl-space-2);
            padding: var(--sl-space-1) var(--sl-space-3);
            background: var(--sl-bg-container);
            border: 1px solid var(--sl-border);
            border-radius: var(--sl-radius-full);
            font-size: 0.7rem;
            color: var(--sl-text-secondary);
        }}

        .pulse-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--sl-success);
            box-shadow: 0 0 8px var(--sl-success);
            animation: pulse 2s ease-in-out infinite;
        }}

        .pulse-dot.offline {{ background: var(--sl-error); box-shadow: 0 0 8px var(--sl-error); animation: none; }}
        .pulse-dot.warning {{ background: var(--sl-warning); box-shadow: 0 0 8px var(--sl-warning); }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
        }}

        /* ─── Grid System ──────────────────────────────────────────── */
        .grid {{
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: var(--sl-space-4);
        }}

        .col-12 {{ grid-column: span 12; }}
        .col-8 {{ grid-column: span 8; }}
        .col-6 {{ grid-column: span 6; }}
        .col-4 {{ grid-column: span 4; }}
        .col-3 {{ grid-column: span 3; }}

        @media (max-width: 1024px) {{
            .col-8, .col-6 {{ grid-column: span 12; }}
            .col-4, .col-3 {{ grid-column: span 6; }}
        }}
        @media (max-width: 640px) {{
            .col-4, .col-3 {{ grid-column: span 12; }}
        }}

        /* ─── Cards (M3 Surface + Glassmorphism) ───────────────────── */
        .card {{
            background: var(--sl-glass-bg);
            backdrop-filter: blur(var(--sl-glass-blur)) saturate(var(--sl-glass-saturate));
            -webkit-backdrop-filter: blur(var(--sl-glass-blur)) saturate(var(--sl-glass-saturate));
            border: 1px solid var(--sl-glass-border);
            border-radius: var(--sl-radius-lg);
            padding: var(--sl-space-5);
            transition: border-color var(--sl-duration-medium) var(--sl-ease-standard),
                        box-shadow var(--sl-duration-medium) var(--sl-ease-standard);
        }}

        .card:hover {{
            border-color: var(--sl-border-variant);
            box-shadow: var(--sl-elevation-2);
        }}

        .card-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: var(--sl-space-4);
        }}

        .card-title {{
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--sl-text-tertiary);
        }}

        .card-action {{
            background: none;
            border: 1px solid var(--sl-border);
            border-radius: var(--sl-radius-sm);
            padding: var(--sl-space-1) var(--sl-space-3);
            color: var(--sl-text-tertiary);
            font-size: 0.65rem;
            cursor: pointer;
            transition: all var(--sl-duration-short) var(--sl-ease-standard);
            font-family: inherit;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}

        .card-action:hover {{
            color: var(--sl-accent);
            border-color: var(--sl-accent-dim);
        }}

        /* ─── Schematic Hero Widget (Spec 012) ─────────────────────── */
        .schematic-hero {{
            width: 100%;
            height: clamp(280px, 45vh, 450px);
            background: linear-gradient(135deg, rgba(13,27,42,0.95) 0%, rgba(13,27,42,0.8) 100%);
            border: 1px solid var(--sl-border);
            border-radius: var(--sl-radius-lg);
            overflow: hidden;
            position: relative;
            margin-bottom: var(--sl-space-5);
        }}

        .schematic-hero::before {{
            content: '';
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(184,115,51,0.03) 1px, transparent 1px) 0 0 / 40px 40px,
                linear-gradient(rgba(184,115,51,0.03) 1px, transparent 1px) 0 0 / 40px 40px;
            pointer-events: none;
        }}

        .schematic-overlay {{
            position: absolute;
            top: var(--sl-space-4);
            left: var(--sl-space-4);
            right: var(--sl-space-4);
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            pointer-events: none;
            z-index: 2;
        }}

        .schematic-title {{
            font-family: var(--sl-font-sans);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--sl-text-primary);
        }}

        .schematic-live-badge {{
            display: flex;
            align-items: center;
            gap: var(--sl-space-2);
            padding: var(--sl-space-1) var(--sl-space-3);
            background: rgba(34, 197, 94, 0.15);
            border-radius: 9999px;
            font-family: var(--sl-font-mono);
            font-size: 0.6rem;
            color: #22C55E;
            pointer-events: auto;
        }}

        .schematic-live-dot {{
            width: 6px;
            height: 6px;
            background: #22C55E;
            border-radius: 50%;
            animation: schematic-pulse 2s infinite;
        }}

        @keyframes schematic-pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.5; transform: scale(0.85); }}
        }}

        .schematic-content {{
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: var(--sl-space-6);
            padding-top: calc(var(--sl-space-6) + 32px);
        }}

        .schematic-content svg {{
            width: 100%;
            height: 100%;
            max-height: 100%;
        }}

        .schematic-loading {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: var(--sl-space-3);
            color: var(--sl-text-disabled);
            font-size: 0.7rem;
        }}

        .schematic-loading-spinner {{
            width: 32px;
            height: 32px;
            border: 2px solid var(--sl-border);
            border-top-color: var(--sl-accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        /* Responsive schematic adjustments */
        @media (max-width: 1199px) {{
            .schematic-hero {{ height: clamp(220px, 35vh, 350px); }}
        }}

        @media (max-width: 767px) {{
            .schematic-hero {{
                height: 180px;
                border-radius: var(--sl-radius-md);
            }}
            .schematic-title {{ font-size: 0.65rem; }}
        }}

        /* ─── Schematic Widget Library (Spec 012 Phase 2) ─────────── */
        /* Modified: 2026-02-09T06:00:00Z | Author: COPILOT | Change: Add Phase 2 widget library CSS */

        /* Compact sidebar schematic widget */
        .schematic-compact {{
            width: 100%;
            aspect-ratio: 16 / 10;
            background: var(--sl-surface-2, #1A1816);
            border: 1px solid var(--sl-border, #2A2624);
            border-radius: var(--sl-radius-md, 8px);
            padding: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}

        .schematic-compact:hover {{
            border-color: var(--sl-accent, #B85A3C);
            box-shadow: 0 0 12px rgba(184, 90, 60, 0.15);
        }}

        .schematic-compact svg {{
            width: 100%;
            height: auto;
            opacity: 0.85;
            transition: opacity 0.2s ease;
        }}

        .schematic-compact:hover svg {{
            opacity: 1;
        }}

        .schematic-compact-label {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 6px;
            font-size: 0.6rem;
            color: var(--sl-text-disabled);
        }}

        .schematic-compact-label .sc-dot {{
            width: 5px;
            height: 5px;
            background: var(--sl-success, #22C55E);
            border-radius: 50%;
            animation: schematic-pulse 2s infinite;
        }}

        /* Card schematic component */
        .card-schematic {{
            background: var(--sl-surface-2, #141211);
            border-radius: var(--sl-radius-md, 8px);
            padding: 12px;
            margin-bottom: 12px;
            position: relative;
            overflow: hidden;
            cursor: pointer;
            transition: background 0.2s ease;
        }}

        .card-schematic:hover {{
            background: var(--sl-surface-3, #1E1C1A);
        }}

        .card-schematic svg {{
            width: 100%;
            height: auto;
            max-height: 200px;
        }}

        .card-schematic .cs-expand {{
            position: absolute;
            top: 8px;
            right: 8px;
            width: 24px;
            height: 24px;
            background: rgba(0,0,0,0.6);
            border: 1px solid var(--sl-border);
            border-radius: 4px;
            color: var(--sl-text-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            cursor: pointer;
            opacity: 0;
            transition: opacity 0.2s ease;
        }}

        .card-schematic:hover .cs-expand {{
            opacity: 1;
        }}

        /* Schematic modal detail view */
        .schematic-modal-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(12px);
            z-index: 3000;
            display: none;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}

        .schematic-modal-overlay.active {{
            display: flex;
            opacity: 1;
        }}

        .schematic-modal {{
            width: 90vw;
            max-width: 1400px;
            height: 80vh;
            background: var(--sl-surface-1, #0F0E0D);
            border: 1px solid var(--sl-border, #2A2624);
            border-radius: var(--sl-radius-lg, 16px);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.5);
        }}

        .schematic-modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            background: var(--sl-surface-2, #1A1816);
            border-bottom: 1px solid var(--sl-border);
        }}

        .schematic-modal-title {{
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--sl-text-primary, #E8E2DE);
        }}

        .schematic-modal-actions {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}

        .schematic-modal-close {{
            width: 32px;
            height: 32px;
            background: transparent;
            border: 1px solid var(--sl-border);
            border-radius: 6px;
            color: var(--sl-text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: background 0.2s ease;
        }}

        .schematic-modal-close:hover {{
            background: rgba(255,255,255,0.05);
        }}

        .schematic-modal-body {{
            flex: 1;
            padding: 24px;
            overflow: auto;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .schematic-modal-body svg {{
            width: 100%;
            height: 100%;
            max-width: 100%;
            max-height: 100%;
        }}

        .schematic-modal-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 24px;
            background: var(--sl-surface-2, #1A1816);
            border-top: 1px solid var(--sl-border);
            font-size: 0.65rem;
            color: var(--sl-text-disabled);
        }}

        /* Status overlay system */
        .schematic-status-overlay {{
            position: absolute;
            bottom: 12px;
            left: 12px;
            right: 12px;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            pointer-events: none;
        }}

        .schematic-status-chip {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 3px 10px;
            background: rgba(0, 0, 0, 0.7);
            border: 1px solid var(--sl-border);
            border-radius: 9999px;
            font-family: 'Consolas', monospace;
            font-size: 10px;
            color: var(--sl-text-secondary);
            pointer-events: auto;
            backdrop-filter: blur(8px);
        }}

        .schematic-status-chip .ssc-dot {{
            width: 5px;
            height: 5px;
            border-radius: 50%;
            flex-shrink: 0;
        }}

        .schematic-status-chip .ssc-dot.active {{ background: var(--sl-success, #22C55E); }}
        .schematic-status-chip .ssc-dot.warning {{ background: var(--sl-warning, #F59E0B); }}
        .schematic-status-chip .ssc-dot.error {{ background: var(--sl-error, #EF4444); }}
        .schematic-status-chip .ssc-dot.idle {{ background: var(--sl-text-disabled, #666); }}

        /* ─── Schematic Interactive Features (Spec 012 Phase 3) ───────── */

        /* Component hover tooltips */
        .schematic-tooltip {{
            position: fixed;
            z-index: 4000;
            padding: 8px 12px;
            background: rgba(15, 14, 13, 0.95);
            border: 1px solid var(--sl-border, #2A2624);
            border-radius: 6px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            font-size: 11px;
            color: var(--sl-text-primary, #E8E2DE);
            pointer-events: none;
            opacity: 0;
            transform: translateY(4px);
            transition: opacity 0.15s ease, transform 0.15s ease;
            max-width: 240px;
        }}

        .schematic-tooltip.visible {{
            opacity: 1;
            transform: translateY(0);
        }}

        .schematic-tooltip-title {{
            font-weight: 600;
            color: var(--sl-accent, #B85A3C);
            margin-bottom: 4px;
        }}

        .schematic-tooltip-status {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-top: 6px;
            padding-top: 6px;
            border-top: 1px solid var(--sl-border, #2A2624);
            font-family: 'Consolas', monospace;
            font-size: 10px;
        }}

        .schematic-tooltip-status .sts-dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }}

        /* Click-to-focus zoom */
        .schematic-zoomable {{
            cursor: zoom-in;
            transition: transform 0.3s var(--sl-ease-standard, cubic-bezier(0.4, 0, 0.2, 1));
            transform-origin: center center;
        }}

        .schematic-zoomable.zoomed {{
            cursor: zoom-out;
        }}

        .schematic-zoom-controls {{
            position: absolute;
            bottom: 16px;
            right: 16px;
            display: flex;
            gap: 4px;
            z-index: 10;
        }}

        .schematic-zoom-btn {{
            width: 32px;
            height: 32px;
            background: rgba(15, 14, 13, 0.85);
            border: 1px solid var(--sl-border, #2A2624);
            border-radius: 6px;
            color: var(--sl-text-secondary, #CAC4BF);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            transition: background 0.2s ease, color 0.2s ease;
        }}

        .schematic-zoom-btn:hover {{
            background: rgba(184, 90, 60, 0.2);
            color: var(--sl-accent, #B85A3C);
        }}

        /* Connection highlighting */
        .schematic-highlight-connections {{
            --highlight-color: var(--sl-accent, #B85A3C);
        }}

        .schematic-connection {{
            transition: stroke 0.2s ease, stroke-width 0.2s ease, filter 0.2s ease;
        }}

        .schematic-connection.highlighted {{
            stroke: var(--highlight-color) !important;
            stroke-width: 3px !important;
            filter: drop-shadow(0 0 6px var(--highlight-color));
        }}

        .schematic-node {{
            transition: filter 0.2s ease, transform 0.2s ease;
        }}

        .schematic-node.highlighted {{
            filter: drop-shadow(0 0 8px var(--sl-accent, #B85A3C));
            transform: scale(1.02);
        }}

        .schematic-node.dimmed {{
            opacity: 0.4;
        }}

        /* Real-time activity flow animation */
        .schematic-flow-active {{
            stroke-dasharray: 6 3;
            animation: schematic-flow 1s linear infinite;
        }}

        @keyframes schematic-flow {{
            to {{ stroke-dashoffset: -9; }}
        }}

        @media (max-width: 767px) {{
            .schematic-compact {{ aspect-ratio: auto; height: 120px; }}
            .schematic-modal {{ width: 96vw; height: 90vh; }}
            .schematic-modal-header {{ padding: 12px 16px; }}
            .schematic-modal-body {{ padding: 12px; }}
        }}

        /* ═══ Phase 3: Watchmaker 3D Dashboard ═══════════════════════════ */
        /* Modified: 2026-02-08T06:00:00Z | Author: COPILOT | Change: Add 3D perspective container, z-layer management, watchmaker views */

        /* 3D Perspective Container */
        .dashboard-3d {{
            perspective: 1200px;
            perspective-origin: 50% 50%;
            transform-style: preserve-3d;
        }}

        /* Z-Layer System */
        .z-background {{
            transform: translateZ(-200px) scale(1.17);
            pointer-events: none;
        }}
        .z-grid {{
            transform: translateZ(-100px) scale(1.083);
            pointer-events: none;
        }}
        .z-connections {{
            transform: translateZ(-50px) scale(1.042);
            pointer-events: none;
        }}
        .z-components {{
            transform: translateZ(0px);
        }}
        .z-floating {{
            transform: translateZ(50px);
        }}
        .z-overlay {{
            transform: translateZ(100px);
        }}

        /* Gear Background Decoration */
        .gear-bg {{
            position: fixed;
            top: 0;
            left: 220px;
            right: 0;
            bottom: 0;
            pointer-events: none;
            overflow: hidden;
            z-index: 0;
            opacity: 0.08;
        }}
        .gear-bg svg {{
            width: 100%;
            height: 100%;
        }}

        /* Watchmaker Card */
        .watchmaker-card {{
            background: var(--sl-bg-container);
            border: 1px solid var(--sl-border);
            border-radius: 16px;
            padding: 32px;
            transform: translateZ(0);
            transform-style: preserve-3d;
            transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease;
            background-image: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0) 50%);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15), 0 1px 2px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }}
        .watchmaker-card:hover {{
            transform: translateZ(12px) rotateX(-1deg);
            box-shadow: 0 12px 24px rgba(0,0,0,0.2), 0 4px 8px rgba(0,0,0,0.15);
        }}
        .watchmaker-card-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 16px;
        }}
        .watchmaker-card-header .gear-icon {{
            width: 24px;
            height: 24px;
            opacity: 0.5;
            flex-shrink: 0;
        }}
        .watchmaker-card.active .gear-icon {{
            animation: gear-spin 8s linear infinite;
        }}
        @keyframes gear-spin {{
            to {{ transform: rotate(360deg); }}
        }}

        /* Status Jewel */
        .status-jewel {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            flex-shrink: 0;
        }}
        .status-jewel.active {{
            background: radial-gradient(circle at 30% 30%, #22C55E 0%, rgba(34,197,94,0.6) 100%);
            box-shadow: 0 0 4px #22C55E, inset 0 1px 2px rgba(255,255,255,0.3);
        }}
        .status-jewel.pending {{
            background: radial-gradient(circle at 30% 30%, #F59E0B 0%, rgba(245,158,11,0.6) 100%);
            box-shadow: 0 0 4px #F59E0B, inset 0 1px 2px rgba(255,255,255,0.3);
            animation: jewel-pulse 1.5s ease-in-out infinite;
        }}
        .status-jewel.error {{
            background: radial-gradient(circle at 30% 30%, #EF4444 0%, rgba(239,68,68,0.6) 100%);
            box-shadow: 0 0 4px #EF4444, inset 0 1px 2px rgba(255,255,255,0.3);
        }}
        .status-jewel.inactive {{
            background: radial-gradient(circle at 30% 30%, #6B7280 0%, rgba(107,114,128,0.6) 100%);
            box-shadow: inset 0 1px 2px rgba(255,255,255,0.2);
        }}
        @keyframes jewel-pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.7; transform: scale(1.15); }}
        }}

        /* Data Flow Line */
        .data-flow-line {{
            stroke: var(--sl-accent, #B85A3C);
            stroke-width: 2;
            stroke-dasharray: 10 5;
            animation: flow-pulse 1.5s linear infinite;
        }}
        @keyframes flow-pulse {{
            to {{ stroke-dashoffset: -15; }}
        }}

        /* System Health Ring */
        .health-ring {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 32px;
            padding: 24px;
            flex-wrap: wrap;
        }}
        .health-ring-node {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            padding: 16px;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            min-width: 100px;
            text-align: center;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }}
        .health-ring-node:hover {{
            border-color: var(--sl-accent);
            box-shadow: 0 0 12px rgba(184,90,60,0.15);
        }}
        .health-ring-node .hr-icon {{
            font-size: 1.4rem;
        }}
        .health-ring-node .hr-label {{
            font-size: 0.65rem;
            color: var(--sl-text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .health-ring-node .hr-value {{
            font-size: 0.85rem;
            font-weight: 700;
            font-family: var(--sl-font-mono);
        }}
        .health-ring-connector {{
            width: 32px;
            height: 2px;
            background: var(--sl-border);
            position: relative;
        }}
        .health-ring-connector::after {{
            content: '';
            position: absolute;
            top: -1px;
            left: 0;
            width: 100%;
            height: 4px;
            background: var(--sl-accent);
            opacity: 0;
            animation: connector-pulse 3s ease-in-out infinite;
        }}
        @keyframes connector-pulse {{
            0%, 100% {{ opacity: 0; }}
            50% {{ opacity: 0.6; }}
        }}

        /* Service Constellation */
        .constellation-view {{
            position: relative;
            min-height: 400px;
            background: rgba(15,14,13,0.4);
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.04);
            overflow: hidden;
        }}
        .constellation-view svg {{
            width: 100%;
            height: 100%;
        }}
        .constellation-legend {{
            display: flex;
            gap: 16px;
            padding: 12px 16px;
            border-top: 1px solid rgba(255,255,255,0.04);
            font-size: 0.6rem;
            color: var(--sl-text-disabled);
        }}
        .constellation-legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        /* GPU Workbench */
        .gpu-workbench {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        .gpu-unit {{
            background: var(--sl-bg-container);
            border: 1px solid var(--sl-border);
            border-radius: 12px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }}
        .gpu-unit-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }}
        .gpu-unit-title {{
            font-weight: 600;
            font-size: 0.85rem;
            color: var(--sl-text-primary);
        }}
        .gpu-meters {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .gpu-meter-row {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .gpu-meter-label {{
            font-size: 0.6rem;
            color: var(--sl-text-secondary);
            min-width: 60px;
            font-family: var(--sl-font-mono);
        }}
        .gpu-meter-bar {{
            flex: 1;
            height: 8px;
            background: rgba(255,255,255,0.05);
            border-radius: 4px;
            overflow: hidden;
        }}
        .gpu-meter-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
            background: linear-gradient(90deg, var(--sl-accent), rgba(184,90,60,0.7));
        }}
        .gpu-meter-value {{
            font-size: 0.65rem;
            font-weight: 600;
            font-family: var(--sl-font-mono);
            min-width: 36px;
            text-align: right;
        }}
        .gpu-tasks-list {{
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid rgba(255,255,255,0.04);
        }}
        .gpu-task-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 4px 0;
            font-size: 0.65rem;
            color: var(--sl-text-secondary);
            font-family: var(--sl-font-mono);
        }}
        .gpu-load-scheduler {{
            grid-column: 1 / -1;
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px;
            background: var(--sl-bg-container);
            border: 1px solid var(--sl-border);
            border-radius: 8px;
            font-size: 0.7rem;
        }}
        .gpu-load-bar {{
            flex: 1;
            height: 12px;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }}
        .gpu-load-fill-0 {{
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            background: linear-gradient(90deg, var(--sl-accent), rgba(184,90,60,0.4));
            border-radius: 6px 0 0 6px;
            transition: width 0.5s ease;
        }}
        .gpu-load-fill-1 {{
            position: absolute;
            right: 0;
            top: 0;
            height: 100%;
            background: linear-gradient(270deg, #98C1D9, rgba(152,193,217,0.4));
            border-radius: 0 6px 6px 0;
            transition: width 0.5s ease;
        }}

        /* Task Orchestration View */
        .task-orchestration {{
            position: relative;
        }}
        .task-pipeline-viz {{
            display: flex;
            align-items: stretch;
            gap: 0;
            padding: 24px 16px;
            overflow-x: auto;
        }}
        .task-pipeline-stage {{
            flex: 1;
            min-width: 140px;
            text-align: center;
            padding: 16px 12px;
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            position: relative;
            background: var(--sl-bg-container);
        }}
        .task-pipeline-stage.current {{
            border-color: var(--sl-accent);
            box-shadow: 0 0 12px rgba(184,90,60,0.15);
        }}
        .task-pipeline-stage-name {{
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }}
        .task-pipeline-stage-count {{
            font-size: 1.4rem;
            font-weight: 700;
            font-family: var(--sl-font-mono);
        }}
        .task-pipeline-arrow {{
            display: flex;
            align-items: center;
            padding: 0 4px;
            color: var(--sl-text-disabled);
            font-size: 1.2rem;
        }}

        @media (max-width: 900px) {{
            .gpu-workbench {{ grid-template-columns: 1fr; }}
            .health-ring {{ gap: 16px; }}
            .health-ring-connector {{ width: 16px; }}
            .task-pipeline-viz {{ flex-direction: column; gap: 8px; }}
            .task-pipeline-arrow {{ transform: rotate(90deg); justify-content: center; }}
            .gear-bg {{ left: 56px; }}
        }}

        /* ═══ Phase 4: Animation System ═════════════════════════════════ */
        /* Modified: 2026-02-08T08:00:00Z | Author: COPILOT | Change: Add gear rotation controller, flow line pulse, jewel pulse variants, card hover depth, background gear parallax */

        /* --- Gear Rotation Speed Tiers (driven by system load) --- */
        .gear-speed-idle {{
            animation-duration: 30s !important;
        }}
        .gear-speed-low {{
            animation-duration: 20s !important;
        }}
        .gear-speed-medium {{
            animation-duration: 10s !important;
        }}
        .gear-speed-high {{
            animation-duration: 5s !important;
        }}
        .gear-speed-max {{
            animation-duration: 2s !important;
        }}
        .gear-paused {{
            animation-play-state: paused !important;
        }}

        /* Counter-rotating gear pairs */
        .gear-reverse {{
            animation-direction: reverse;
        }}
        @keyframes gear-spin-eased {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        /* Background gear parallax containers */
        .gear-bg .gear-parallax-layer {{
            position: absolute;
            transition: transform 0.3s ease-out;
            will-change: transform;
        }}
        .gear-parallax-far {{
            opacity: 0.03;
        }}
        .gear-parallax-mid {{
            opacity: 0.06;
        }}
        .gear-parallax-near {{
            opacity: 0.1;
        }}

        /* --- Enhanced Flow Line Variants --- */
        .data-flow-line.flow-fast {{
            animation-duration: 0.6s;
            stroke-width: 3;
        }}
        .data-flow-line.flow-medium {{
            animation-duration: 1.5s;
            stroke-width: 2;
        }}
        .data-flow-line.flow-slow {{
            animation-duration: 3s;
            stroke-width: 1.5;
            opacity: 0.6;
        }}
        .data-flow-line.flow-inactive {{
            animation-play-state: paused;
            opacity: 0.2;
            stroke-dasharray: 4 8;
        }}
        @keyframes flow-pulse-glow {{
            0% {{ stroke-dashoffset: 0; filter: drop-shadow(0 0 0px var(--sl-accent, #B85A3C)); }}
            50% {{ filter: drop-shadow(0 0 4px var(--sl-accent, #B85A3C)); }}
            100% {{ stroke-dashoffset: -15; filter: drop-shadow(0 0 0px var(--sl-accent, #B85A3C)); }}
        }}
        .data-flow-line.flow-glow {{
            animation-name: flow-pulse-glow;
        }}

        /* --- Jewel Pulse State Variants --- */
        @keyframes jewel-pulse-urgent {{
            0%, 100% {{ opacity: 1; transform: scale(1); box-shadow: 0 0 4px currentColor; }}
            50% {{ opacity: 0.8; transform: scale(1.3); box-shadow: 0 0 12px currentColor; }}
        }}
        @keyframes jewel-pulse-slow {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.85; transform: scale(1.05); }}
        }}
        @keyframes jewel-breathe {{
            0%, 100% {{ opacity: 0.7; }}
            50% {{ opacity: 1; }}
        }}
        .status-jewel.pulse-urgent {{
            animation: jewel-pulse-urgent 0.8s ease-in-out infinite;
        }}
        .status-jewel.pulse-slow {{
            animation: jewel-pulse-slow 3s ease-in-out infinite;
        }}
        .status-jewel.breathe {{
            animation: jewel-breathe 4s ease-in-out infinite;
        }}

        /* --- Card Hover Depth System --- */
        .watchmaker-card {{
            --card-rx: 0deg;
            --card-ry: 0deg;
            --card-tz: 0px;
            --card-shadow-x: 0px;
            --card-shadow-y: 4px;
        }}
        .watchmaker-card.depth-tracking {{
            transform: translateZ(var(--card-tz)) rotateX(var(--card-rx)) rotateY(var(--card-ry));
            box-shadow:
                calc(var(--card-shadow-x) * -1) calc(var(--card-shadow-y) + 4px) 24px rgba(0,0,0,0.2),
                0 2px 8px rgba(0,0,0,0.12);
            transition: box-shadow 0.15s ease;
        }}
        .watchmaker-card.depth-lift {{
            --card-tz: 16px;
            box-shadow:
                0 16px 32px rgba(0,0,0,0.25),
                0 4px 12px rgba(0,0,0,0.15),
                inset 0 1px 0 rgba(255,255,255,0.06);
        }}

        /* Polish shimmer on hover */
        @keyframes card-shimmer {{
            0% {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}
        .watchmaker-card.shimmer-active::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: inherit;
            pointer-events: none;
            background: linear-gradient(90deg,
                transparent 0%,
                rgba(255,255,255,0.04) 45%,
                rgba(255,255,255,0.08) 50%,
                rgba(255,255,255,0.04) 55%,
                transparent 100%
            );
            background-size: 200% 100%;
            animation: card-shimmer 2s ease-in-out;
        }}

        /* --- Background Gear Parallax Layers --- */
        .gear-bg .gear-layer-1 {{
            position: absolute;
            top: 10%;
            left: 15%;
            width: 180px;
            height: 180px;
        }}
        .gear-bg .gear-layer-2 {{
            position: absolute;
            top: 50%;
            right: 10%;
            width: 120px;
            height: 120px;
        }}
        .gear-bg .gear-layer-3 {{
            position: absolute;
            bottom: 15%;
            left: 40%;
            width: 220px;
            height: 220px;
        }}
        .gear-bg .gear-layer-4 {{
            position: absolute;
            top: 25%;
            right: 35%;
            width: 80px;
            height: 80px;
        }}
        .gear-bg .gear-layer-5 {{
            position: absolute;
            bottom: 40%;
            left: 5%;
            width: 100px;
            height: 100px;
        }}
        .gear-bg svg.gear-animated {{
            animation: gear-spin-eased 20s linear infinite;
        }}
        .gear-bg svg.gear-animated.reverse {{
            animation-direction: reverse;
        }}

        /* Reduced motion: disable all watchmaker animations */
        @media (prefers-reduced-motion: reduce) {{
            .gear-bg svg.gear-animated,
            .watchmaker-card.active .gear-icon,
            .status-jewel.pending,
            .status-jewel.pulse-urgent,
            .status-jewel.pulse-slow,
            .status-jewel.breathe,
            .data-flow-line,
            .health-ring-connector::after,
            .watchmaker-card.shimmer-active::after {{
                animation: none !important;
            }}
            .watchmaker-card.depth-tracking {{
                transform: none !important;
            }}
        }}

        /* ─── Buttons (M3-inspired) ────────────────────────────────── */
        .btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: var(--sl-space-2);
            padding: var(--sl-space-2) var(--sl-space-4);
            border-radius: var(--sl-radius-sm);
            font-family: inherit;
            font-size: 0.75rem;
            font-weight: 500;
            cursor: pointer;
            border: none;
            transition: all var(--sl-duration-short) var(--sl-ease-standard);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        .btn-filled {{
            background: var(--sl-accent);
            color: var(--sl-text-on-accent);
        }}
        .btn-filled:hover {{
            box-shadow: var(--sl-elevation-2);
            filter: brightness(0.9);
        }}

        .btn-tonal {{
            background: var(--sl-accent-container);
            color: var(--sl-accent-on-container);
            border: 1px solid var(--sl-accent-dim);
        }}
        .btn-tonal:hover {{
            background: rgba(184,115,51,0.18);
        }}

        .btn-outline {{
            background: transparent;
            border: 1px solid var(--sl-border);
            color: var(--sl-text-secondary);
        }}
        .btn-outline:hover {{
            border-color: var(--sl-accent-dim);
            color: var(--sl-accent);
            background: rgba(184,115,51,0.06);
        }}

        .btn-ghost {{
            background: transparent;
            border: none;
            color: var(--sl-text-tertiary);
        }}
        .btn-ghost:hover {{
            color: var(--sl-text-primary);
            background: rgba(255,255,255,0.03);
        }}

        /* ─── SLATE Control Panel ──────────────────────────────────── */
        .ctrl-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: var(--sl-space-3);
        }}
        @media (max-width: 900px) {{ .ctrl-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
        @media (max-width: 540px) {{ .ctrl-grid {{ grid-template-columns: 1fr; }} }}

        .ctrl-btn {{
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: var(--sl-space-2);
            padding: var(--sl-space-5) var(--sl-space-3);
            border-radius: var(--sl-radius-md);
            background: var(--sl-bg-container);
            border: 1px solid var(--sl-border);
            color: var(--sl-text-secondary);
            cursor: pointer;
            transition: all var(--sl-duration-medium) var(--sl-ease-emphasized);
            font-family: inherit;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            overflow: hidden;
        }}

        .ctrl-btn::before {{
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(184,115,51,0.05) 0%, transparent 60%);
            opacity: 0;
            transition: opacity var(--sl-duration-medium);
        }}

        .ctrl-btn:hover {{
            border-color: var(--sl-accent-dim);
            background: var(--sl-bg-container-high);
            transform: translateY(-2px);
            box-shadow: var(--sl-elevation-3);
        }}
        .ctrl-btn:hover::before {{ opacity: 1; }}
        .ctrl-btn:active {{ transform: translateY(0); }}

        .ctrl-btn.running {{
            border-color: var(--sl-accent);
            pointer-events: none;
        }}
        .ctrl-btn.running .ctrl-icon {{
            animation: spin 1.2s linear infinite;
        }}
        .ctrl-btn.ok {{ border-color: var(--sl-success); }}
        .ctrl-btn.err {{ border-color: var(--sl-error); }}

        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

        .ctrl-icon {{
            font-size: 1.5rem;
            line-height: 1;
            position: relative;
            z-index: 1;
        }}

        .ctrl-label {{ position: relative; z-index: 1; }}

        .ctrl-dot {{
            position: absolute;
            top: var(--sl-space-2);
            right: var(--sl-space-2);
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--sl-text-disabled);
        }}
        .ctrl-dot.ok {{ background: var(--sl-success); box-shadow: 0 0 6px var(--sl-success); }}
        .ctrl-dot.err {{ background: var(--sl-error); box-shadow: 0 0 6px var(--sl-error); }}
        .ctrl-dot.warn {{ background: var(--sl-warning); box-shadow: 0 0 6px var(--sl-warning); }}

        /* Control output terminal */
        .ctrl-term {{
            display: none;
            margin-top: var(--sl-space-4);
            padding: var(--sl-space-4);
            background: rgba(0,0,0,0.5);
            border: 1px solid var(--sl-border);
            border-radius: var(--sl-radius-sm);
            font-family: var(--sl-font-mono);
            font-size: 0.7rem;
            line-height: 1.7;
            color: var(--sl-text-tertiary);
            max-height: 280px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .ctrl-term.open {{ display: block; animation: fadeIn 0.2s ease; }}
        .ctrl-term .ok {{ color: var(--sl-success); }}
        .ctrl-term .err {{ color: var(--sl-error); }}
        .ctrl-term .lbl {{ color: var(--sl-text-secondary); font-weight: 600; }}

        /* CI dispatch row */
        .ci-row {{
            display: flex;
            gap: var(--sl-space-2);
            flex-wrap: wrap;
            margin-top: var(--sl-space-4);
            padding-top: var(--sl-space-4);
            border-top: 1px solid var(--sl-border);
        }}
        .ci-row .btn {{ flex: 1; min-width: 100px; font-size: 0.65rem; }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* ─── Pipeline Visualization ───────────────────────────────── */
        .pipeline {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0;
            padding: var(--sl-space-4) 0;
        }}

        .pipe-node {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: var(--sl-space-1);
        }}

        .pipe-icon {{
            width: 44px;
            height: 44px;
            border-radius: var(--sl-radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.85rem;
            border: 1px solid var(--sl-border);
            background: var(--sl-bg-container);
            color: var(--sl-text-secondary);
            transition: all var(--sl-duration-medium) var(--sl-ease-standard);
        }}

        .pipe-icon.active {{
            border-color: var(--sl-accent);
            color: var(--sl-accent);
            box-shadow: 0 0 16px var(--sl-accent-glow);
        }}

        .pipe-label {{
            font-size: 0.6rem;
            color: var(--sl-text-disabled);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}

        .pipe-value {{
            font-size: 0.65rem;
            color: var(--sl-text-tertiary);
            font-family: var(--sl-font-mono);
        }}

        .pipe-connector {{
            width: 48px;
            height: 1px;
            background: var(--sl-border);
            margin: 0 var(--sl-space-1);
            margin-bottom: 28px;
        }}
        .pipe-connector.active {{
            background: var(--sl-accent-dim);
            box-shadow: 0 0 4px var(--sl-accent-glow);
        }}

        /* ─── Metrics & Progress Bars ──────────────────────────────── */
        .metric {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: var(--sl-space-2) 0;
        }}
        .metric + .metric {{ border-top: 1px solid rgba(255,255,255,0.03); }}

        .metric-label {{
            font-size: 0.7rem;
            color: var(--sl-text-tertiary);
        }}

        .metric-bar {{
            flex: 1;
            margin: 0 var(--sl-space-3);
            height: 4px;
            background: var(--sl-bg-container-highest);
            border-radius: var(--sl-radius-full);
            overflow: hidden;
        }}

        .metric-fill {{
            height: 100%;
            background: var(--sl-accent);
            border-radius: var(--sl-radius-full);
            transition: width var(--sl-duration-long) var(--sl-ease-decelerate);
        }}

        .metric-value {{
            font-size: 0.7rem;
            font-family: var(--sl-font-mono);
            color: var(--sl-text-secondary);
            min-width: 36px;
            text-align: right;
        }}

        /* ─── Status Badges ────────────────────────────────────────── */
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 8px;
            border-radius: var(--sl-radius-full);
            font-size: 0.6rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .badge.online {{ background: var(--sl-success-container); color: var(--sl-success); }}
        .badge.offline {{ background: var(--sl-error-container); color: var(--sl-error); }}
        .badge.pending {{ background: rgba(255,255,255,0.05); color: var(--sl-text-disabled); }}
        .badge.warning {{ background: var(--sl-warning-container); color: var(--sl-warning); }}

        /* ─── Service List ─────────────────────────────────────────── */
        .svc-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: var(--sl-space-2) var(--sl-space-3);
            border-radius: var(--sl-radius-sm);
            transition: background var(--sl-duration-short);
        }}
        .svc-item:hover {{ background: rgba(255,255,255,0.02); }}
        .svc-item + .svc-item {{ border-top: 1px solid rgba(255,255,255,0.03); }}

        .svc-name {{
            display: flex;
            align-items: center;
            gap: var(--sl-space-2);
            font-size: 0.75rem;
            color: var(--sl-text-secondary);
        }}

        .svc-icon {{
            width: 28px;
            height: 28px;
            border-radius: var(--sl-radius-sm);
            background: var(--sl-bg-container-high);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--sl-text-tertiary);
        }}

        /* ─── Task List ────────────────────────────────────────────── */
        .task-form {{
            display: flex;
            gap: var(--sl-space-2);
            margin-bottom: var(--sl-space-3);
        }}

        .task-input {{
            flex: 1;
            padding: var(--sl-space-2) var(--sl-space-3);
            background: var(--sl-bg-container);
            border: 1px solid var(--sl-border);
            border-radius: var(--sl-radius-sm);
            color: var(--sl-text-primary);
            font-family: inherit;
            font-size: 0.75rem;
            outline: none;
            transition: border-color var(--sl-duration-short);
        }}
        .task-input:focus {{ border-color: var(--sl-accent); }}
        .task-input::placeholder {{ color: var(--sl-text-disabled); }}

        .task-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: var(--sl-space-2) var(--sl-space-3);
            border-radius: var(--sl-radius-sm);
            font-size: 0.75rem;
            color: var(--sl-text-secondary);
        }}
        .task-item + .task-item {{ border-top: 1px solid rgba(255,255,255,0.03); }}

        /* ─── Workflow List ─────────────────────────────────────────── */
        .wf-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: var(--sl-space-2) 0;
            font-size: 0.7rem;
        }}
        .wf-item + .wf-item {{ border-top: 1px solid rgba(255,255,255,0.03); }}

        .wf-name {{
            color: var(--sl-text-secondary);
            display: flex;
            align-items: center;
            gap: var(--sl-space-2);
        }}

        .wf-meta {{
            display: flex;
            align-items: center;
            gap: var(--sl-space-3);
            color: var(--sl-text-disabled);
            font-family: var(--sl-font-mono);
            font-size: 0.6rem;
        }}

        /* ─── Heatmap ──────────────────────────────────────────────── */
        .heatmap-grid {{
            display: grid;
            grid-template-columns: repeat(52, 1fr);
            gap: 3px;
        }}
        .heatmap-week {{ display: flex; flex-direction: column; gap: 3px; }}
        .heatmap-day {{
            width: 10px; height: 10px;
            border-radius: 2px;
            background: rgba(255,255,255,0.04);
            transition: transform 0.15s;
        }}
        .heatmap-day:hover {{ transform: scale(1.4); }}
        .heatmap-day[data-level="1"] {{ background: rgba(184,115,51,0.2); }}
        .heatmap-day[data-level="2"] {{ background: rgba(184,115,51,0.4); }}
        .heatmap-day[data-level="3"] {{ background: rgba(184,115,51,0.6); }}
        .heatmap-day[data-level="4"] {{ background: rgba(184,115,51,0.85); }}

        /* ─── GitHub Sections ──────────────────────────────────────── */
        .gh-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: var(--sl-space-4);
        }}
        @media (max-width: 900px) {{ .gh-grid {{ grid-template-columns: repeat(2, 1fr); }} }}

        .gh-section-title {{
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--sl-text-disabled);
            margin-bottom: var(--sl-space-3);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .gh-section-title .count {{
            background: var(--sl-bg-container-high);
            padding: 1px 6px;
            border-radius: var(--sl-radius-full);
            font-family: var(--sl-font-mono);
            font-size: 0.6rem;
        }}

        .gh-item {{
            padding: var(--sl-space-2) 0;
            font-size: 0.7rem;
            color: var(--sl-text-tertiary);
            border-bottom: 1px solid rgba(255,255,255,0.02);
        }}

        /* ─── Hardware Meters ──────────────────────────────────────── */
        .hw-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: var(--sl-space-4); }}
        @media (max-width: 640px) {{ .hw-grid {{ grid-template-columns: 1fr; }} }}

        .hw-panel {{
            padding: var(--sl-space-4);
            background: var(--sl-bg-container);
            border-radius: var(--sl-radius-md);
            border: 1px solid rgba(255,255,255,0.03);
        }}

        .hw-title {{
            font-size: 0.65rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--sl-text-disabled);
            margin-bottom: var(--sl-space-3);
        }}

        .hw-stat {{
            text-align: center;
            padding: var(--sl-space-3);
        }}
        .hw-stat-value {{
            font-size: 1.8rem;
            font-weight: 700;
            font-family: var(--sl-font-mono);
            color: var(--sl-accent);
        }}
        .hw-stat-label {{
            font-size: 0.6rem;
            color: var(--sl-text-disabled);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 2px;
        }}

        /* ─── Onboarding Overlay ───────────────────────────────────── */
        .onboard-overlay {{
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.85);
            backdrop-filter: blur(12px);
            z-index: 1000;
            display: none;
            align-items: center;
            justify-content: center;
        }}
        .onboard-overlay.active {{ display: flex; }}

        .onboard-card {{
            max-width: 560px;
            width: 90%;
            background: var(--sl-bg-surface);
            border: 1px solid var(--sl-border);
            border-radius: var(--sl-radius-xl);
            padding: var(--sl-space-10);
            text-align: center;
            animation: fadeIn 0.4s ease;
        }}

        .onboard-card h2 {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--sl-text-primary);
            margin-bottom: var(--sl-space-2);
        }}

        .onboard-card p {{
            color: var(--sl-text-tertiary);
            font-size: 0.85rem;
            margin-bottom: var(--sl-space-6);
            line-height: 1.7;
        }}

        .onboard-steps {{
            display: flex;
            flex-direction: column;
            gap: var(--sl-space-3);
            text-align: left;
            margin-bottom: var(--sl-space-6);
        }}

        .onboard-step {{
            display: flex;
            align-items: flex-start;
            gap: var(--sl-space-3);
            padding: var(--sl-space-3);
            border-radius: var(--sl-radius-sm);
            background: var(--sl-bg-container);
        }}

        .onboard-step-num {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: var(--sl-accent-container);
            color: var(--sl-accent);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            font-weight: 700;
            flex-shrink: 0;
        }}

        .onboard-step-text {{
            font-size: 0.8rem;
            color: var(--sl-text-secondary);
            line-height: 1.5;
        }}

        .onboard-step-text strong {{
            color: var(--sl-accent);
        }}

        /* ─── Empty State ──────────────────────────────────────────── */
        .empty {{
            text-align: center;
            padding: var(--sl-space-8) var(--sl-space-4);
            color: var(--sl-text-disabled);
            font-size: 0.75rem;
        }}

        /* ─── Connection Status ────────────────────────────────────── */
        #conn-status {{
            position: fixed;
            bottom: var(--sl-space-4);
            right: var(--sl-space-4);
            padding: var(--sl-space-1) var(--sl-space-3);
            background: var(--sl-bg-container);
            border: 1px solid var(--sl-border);
            border-radius: var(--sl-radius-full);
            font-size: 0.6rem;
            color: var(--sl-text-disabled);
            z-index: 100;
            display: flex;
            align-items: center;
            gap: var(--sl-space-1);
        }}

        /* ─── Activity Feed ────────────────────────────────────────── */
        .feed-item {{
            display: flex;
            align-items: flex-start;
            gap: var(--sl-space-3);
            padding: var(--sl-space-2) 0;
        }}
        .feed-item + .feed-item {{ border-top: 1px solid rgba(255,255,255,0.02); }}

        .feed-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--sl-accent-dim);
            margin-top: 5px;
            flex-shrink: 0;
        }}

        .feed-text {{
            font-size: 0.7rem;
            color: var(--sl-text-tertiary);
            line-height: 1.5;
        }}

        .feed-time {{
            font-size: 0.6rem;
            color: var(--sl-text-disabled);
            font-family: var(--sl-font-mono);
        }}

        /* ─── Scrollbar ────────────────────────────────────────────── */
        ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.08); border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.15); }}

        /* ─── Skip Link ────────────────────────────────────────────── */
        .skip-link {{
            position: absolute; left: -9999px; top: -9999px;
            padding: var(--sl-space-2) var(--sl-space-4);
            background: var(--sl-accent); color: #000;
            z-index: 9999; font-weight: 600;
        }}
        .skip-link:focus {{ left: 0; top: 0; }}

        /* ─── Nav Sidebar (Menu-Driven) ────────────────────────────── */
        .dashboard-layout {{
            display: flex;
            min-height: 100vh;
        }}

        .nav-sidebar {{
            width: 220px;
            min-height: 100vh;
            background: #060606;
            border-right: 1px solid rgba(255,255,255,0.06);
            position: fixed;
            left: 0;
            top: 0;
            z-index: 50;
            display: flex;
            flex-direction: column;
            padding: var(--sl-space-5) 0;
            overflow-y: auto;
        }}

        .nav-brand {{
            display: flex;
            align-items: center;
            gap: var(--sl-space-3);
            padding: 0 var(--sl-space-5);
            margin-bottom: var(--sl-space-6);
        }}

        .nav-brand-text {{
            display: flex;
            flex-direction: column;
        }}

        .nav-brand-name {{
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            color: var(--sl-text-primary);
        }}

        .nav-brand-sub {{
            font-size: 0.55rem;
            color: var(--sl-text-disabled);
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }}

        .nav-section {{
            padding: 0;
            margin-bottom: var(--sl-space-2);
        }}

        .nav-section-label {{
            font-size: 0.55rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--sl-text-disabled);
            padding: var(--sl-space-2) var(--sl-space-5);
        }}

        .nav-item {{
            display: flex;
            align-items: center;
            gap: var(--sl-space-3);
            padding: var(--sl-space-2) var(--sl-space-5);
            color: var(--sl-text-tertiary);
            cursor: pointer;
            font-size: 0.75rem;
            font-weight: 500;
            transition: all 0.15s ease;
            border-left: 2px solid transparent;
            text-decoration: none;
            background: none;
            border-top: none;
            border-bottom: none;
            border-right: none;
            width: 100%;
            font-family: inherit;
        }}

        .nav-item:hover {{
            color: var(--sl-text-secondary);
            background: rgba(255,255,255,0.02);
        }}

        .nav-item.active {{
            color: var(--sl-accent);
            border-left-color: var(--sl-accent);
            background: rgba(184,115,51,0.06);
        }}

        .nav-item-icon {{
            width: 18px;
            text-align: center;
            font-size: 0.85rem;
        }}

        .nav-item-badge {{
            margin-left: auto;
            background: var(--sl-accent-container);
            color: var(--sl-accent);
            padding: 1px 6px;
            border-radius: var(--sl-radius-full);
            font-size: 0.55rem;
            font-weight: 700;
            font-family: var(--sl-font-mono);
        }}

        .nav-footer {{
            margin-top: auto;
            padding: var(--sl-space-4) var(--sl-space-5);
            border-top: 1px solid rgba(255,255,255,0.04);
        }}

        .nav-footer-status {{
            display: flex;
            align-items: center;
            gap: var(--sl-space-2);
            font-size: 0.6rem;
            color: var(--sl-text-disabled);
        }}

        /* Main content area offset for sidebar */
        .main-content {{
            margin-left: 220px;
            flex: 1;
            min-height: 100vh;
        }}

        /* Dashboard sections (show/hide by nav) */
        .dash-section {{
            display: none;
        }}
        .dash-section.active {{
            display: block;
            animation: fadeIn 0.25s ease;
        }}

        @media (max-width: 900px) {{
            .nav-sidebar {{ width: 56px; overflow: hidden; }}
            .nav-sidebar .nav-brand-text,
            .nav-sidebar .nav-section-label,
            .nav-sidebar .nav-item span:not(.nav-item-icon),
            .nav-sidebar .nav-item-badge,
            .nav-sidebar .nav-footer-status span {{ display: none; }}
            .nav-item {{ padding: var(--sl-space-2) var(--sl-space-3); justify-content: center; }}
            .main-content {{ margin-left: 56px; }}
        }}

        /* ─── Docker Containers ────────────────────────────────────── */
        .docker-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: var(--sl-space-3);
        }}

        /* ─── Tech Tree ────────────────────────────────────────────── */
        .tech-tree-stats {{
            display: flex;
            gap: var(--sl-space-4);
            flex-wrap: wrap;
        }}
        .tt-stat {{
            text-align: center;
            padding: var(--sl-space-2) var(--sl-space-4);
            background: var(--sl-bg-container);
            border-radius: var(--sl-radius-sm);
            border: 1px solid rgba(255,255,255,0.03);
        }}
        .tt-stat-val {{
            font-size: 1.2rem;
            font-weight: 700;
            font-family: var(--sl-font-mono);
            color: var(--sl-accent);
        }}
        .tt-stat-lbl {{
            font-size: 0.55rem;
            color: var(--sl-text-disabled);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}

        /* ─── Select Dropdown ──────────────────────────────────────── */
        select {{
            padding: var(--sl-space-2) var(--sl-space-3);
            background: var(--sl-bg-container);
            border: 1px solid var(--sl-border);
            border-radius: var(--sl-radius-sm);
            color: var(--sl-text-primary);
            font-family: inherit;
            font-size: 0.75rem;
            outline: none;
        }}
        select:focus {{ border-color: var(--sl-accent); }}

    </style>
</head>
<body>
    <a class="skip-link" href="#main">Skip to main content</a>

    <!-- Onboarding Overlay -->
    <div class="onboard-overlay" id="onboard">
        <div class="onboard-card">
            <div style="margin-bottom: 20px;">{logo_icon_svg}</div>
            <h2>Welcome to S.L.A.T.E.</h2>
            <p>Synchronized Living Architecture for Transformation and Evolution.<br>
            Your local AI orchestration system is ready to get started.</p>
            <div class="onboard-steps">
                <div class="onboard-step">
                    <div class="onboard-step-num">1</div>
                    <div class="onboard-step-text"><strong>Quick Health Check</strong> &mdash; Click "Run SLATE" in the Controls panel to verify your system is working. This checks all 7 components automatically.</div>
                </div>
                <div class="onboard-step">
                    <div class="onboard-step-num">2</div>
                    <div class="onboard-step-text"><strong>Navigate with the Sidebar</strong> &mdash; Use the left-hand menu to explore different areas: Controls, Hardware, Workflows, and more.</div>
                </div>
                <div class="onboard-step">
                    <div class="onboard-step-num">3</div>
                    <div class="onboard-step-text"><strong>Ask @slate in VS Code</strong> &mdash; Open VS Code Chat (Ctrl+L) and type <strong>@slate</strong> to interact with the system using natural language.</div>
                </div>
                <div class="onboard-step">
                    <div class="onboard-step-num">4</div>
                    <div class="onboard-step-text"><strong>Apply the Theme</strong> &mdash; Press Ctrl+K then Ctrl+T and search for "SLATE Dark" to match your editor to the dashboard design.</div>
                </div>
            </div>
            <button class="btn btn-filled" onclick="dismissOnboard()">Get Started</button>
        </div>
    </div>

    <div class="dashboard-layout">
        <!-- ═══ Gear Background Parallax (Phase 4) ═══ -->
        <div class="gear-bg" id="gear-bg-container">
            <div class="gear-layer-1 gear-parallax-layer gear-parallax-far">
                <svg class="gear-animated" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                    <g style="transform-origin:50% 50%"><circle cx="50" cy="50" r="28" fill="none" stroke="#B85A3C" stroke-width="3"/>
                    <path d="M50,8 L54,18 L46,18Z M92,50 L82,54 L82,46Z M50,92 L46,82 L54,82Z M8,50 L18,46 L18,54Z M79,21 L72,29 L73,27Z M79,79 L72,72 L27,73Z M21,79 L29,72 L27,73Z M21,21 L29,29 L27,27Z" fill="#B85A3C"/>
                    <circle cx="50" cy="50" r="10" fill="#B85A3C" opacity="0.3"/></g>
                </svg>
            </div>
            <div class="gear-layer-2 gear-parallax-layer gear-parallax-mid">
                <svg class="gear-animated reverse" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                    <g style="transform-origin:50% 50%"><circle cx="50" cy="50" r="24" fill="none" stroke="#98C1D9" stroke-width="2"/>
                    <path d="M50,10 L53,20 L47,20Z M90,50 L80,53 L80,47Z M50,90 L47,80 L53,80Z M10,50 L20,47 L20,53Z M78,22 L72,28Z M78,78 L72,72Z M22,78 L28,72Z M22,22 L28,28Z" fill="#98C1D9"/>
                    <circle cx="50" cy="50" r="8" fill="#98C1D9" opacity="0.2"/></g>
                </svg>
            </div>
            <div class="gear-layer-3 gear-parallax-layer gear-parallax-far">
                <svg class="gear-animated" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="animation-duration:30s;">
                    <g style="transform-origin:50% 50%"><circle cx="50" cy="50" r="32" fill="none" stroke="#B85A3C" stroke-width="2.5"/>
                    <path d="M50,5 L55,17 L45,17Z M95,50 L83,55 L83,45Z M50,95 L45,83 L55,83Z M5,50 L17,45 L17,55Z M82,18 L73,26Z M82,82 L73,74Z M18,82 L26,74Z M18,18 L26,26Z M68,9 L64,20Z M91,32 L80,36Z M91,68 L80,64Z M68,91 L64,80Z M32,91 L36,80Z M9,68 L20,64Z M9,32 L20,36Z M32,9 L36,20Z" fill="#B85A3C" opacity="0.8"/>
                    <circle cx="50" cy="50" r="14" fill="#B85A3C" opacity="0.15"/></g>
                </svg>
            </div>
            <div class="gear-layer-4 gear-parallax-layer gear-parallax-mid">
                <svg class="gear-animated reverse" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="animation-duration:15s;">
                    <g style="transform-origin:50% 50%"><circle cx="50" cy="50" r="22" fill="none" stroke="#98C1D9" stroke-width="2"/>
                    <path d="M50,12 L53,22 L47,22Z M88,50 L78,53 L78,47Z M50,88 L47,78 L53,78Z M12,50 L22,47 L22,53Z" fill="#98C1D9" opacity="0.5"/>
                    <circle cx="50" cy="50" r="7" fill="#98C1D9" opacity="0.15"/></g>
                </svg>
            </div>
            <div class="gear-layer-5 gear-parallax-layer gear-parallax-near">
                <svg class="gear-animated" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="animation-duration:12s;">
                    <g style="transform-origin:50% 50%"><circle cx="50" cy="50" r="20" fill="none" stroke="#B85A3C" stroke-width="2.5"/>
                    <path d="M50,14 L53,24 L47,24Z M86,50 L76,53 L76,47Z M50,86 L47,76 L53,76Z M14,50 L24,47 L24,53Z M74,26 L68,32Z M74,74 L68,68Z M26,74 L32,68Z M26,26 L32,32Z" fill="#B85A3C"/>
                    <circle cx="50" cy="50" r="6" fill="#B85A3C" opacity="0.25"/></g>
                </svg>
            </div>
        </div>

        <!-- ═══ Sidebar Navigation ═══ -->
        <nav class="nav-sidebar" role="navigation" aria-label="Main navigation">
            <div class="nav-brand">
                <div style="width:28px;height:28px;flex-shrink:0;">{logo_icon_svg}</div>
                <div class="nav-brand-text">
                    <span class="nav-brand-name">S.L.A.T.E.</span>
                    <span class="nav-brand-sub">Architecture</span>
                </div>
            </div>

            <div class="nav-section">
                <div class="nav-section-label">Operate</div>
                <button class="nav-item active" onclick="showSection('overview', this)">
                    <span class="nav-item-icon">&#9670;</span><span>Overview</span>
                </button>
                <button class="nav-item" onclick="showSection('controls', this)">
                    <span class="nav-item-icon">&#9881;</span><span>Controls</span>
                </button>
                <button class="nav-item" onclick="showSection('hardware', this)">
                    <span class="nav-item-icon">&#9889;</span><span>Hardware</span>
                </button>
            </div>

            <div class="nav-section">
                <div class="nav-section-label">Build</div>
                <button class="nav-item" onclick="showSection('workflows', this)">
                    <span class="nav-item-icon">&#8635;</span><span>Workflows</span>
                    <span class="nav-item-badge" id="nav-wf-count">0</span>
                </button>
                <button class="nav-item" onclick="showSection('tasks', this)">
                    <span class="nav-item-icon">&#9744;</span><span>Tasks</span>
                    <span class="nav-item-badge" id="nav-task-count">0</span>
                </button>
                <button class="nav-item" onclick="showSection('github', this)">
                    <span class="nav-item-icon">&#128279;</span><span>GitHub</span>
                </button>
            </div>

            <div class="nav-section">
                <div class="nav-section-label">Intelligence</div>
                <button class="nav-item" onclick="showSection('agents', this)">
                    <span class="nav-item-icon">&#129302;</span><span>Agents</span>
                </button>
                <button class="nav-item" onclick="showSection('activity', this)">
                    <span class="nav-item-icon">&#128200;</span><span>Activity</span>
                </button>
            </div>

            <div class="nav-section">
                <div class="nav-section-label">Watchmaker</div>
                <button class="nav-item" onclick="showSection('command-center', this)">
                    <span class="nav-item-icon">&#9881;</span><span>Command</span>
                </button>
                <button class="nav-item" onclick="showSection('constellation', this)">
                    <span class="nav-item-icon">&#9733;</span><span>Services</span>
                </button>
                <button class="nav-item" onclick="showSection('gpu-workbench', this)">
                    <span class="nav-item-icon">&#9883;</span><span>GPU Lab</span>
                </button>
                <button class="nav-item" onclick="showSection('task-orch', this)">
                    <span class="nav-item-icon">&#8631;</span><span>Pipeline</span>
                </button>
            </div>

            <!-- Compact Schematic Widget (Spec 012 Phase 2) -->
            <div style="padding: 0 var(--sl-space-4); margin-bottom: var(--sl-space-4);">
                <div class="schematic-compact" id="sidebar-schematic" onclick="openSchematicModal('system')" title="Click to expand system architecture">
                    <div id="sidebar-schematic-svg">
                        <svg viewBox="0 0 160 100" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;opacity:0.6;">
                            <rect x="5" y="10" width="30" height="20" rx="3" fill="none" stroke="#3A3634" stroke-width="1"/>
                            <rect x="65" y="10" width="30" height="20" rx="3" fill="none" stroke="#3A3634" stroke-width="1"/>
                            <rect x="125" y="10" width="30" height="20" rx="3" fill="none" stroke="#3A3634" stroke-width="1"/>
                            <rect x="35" y="50" width="30" height="20" rx="3" fill="none" stroke="#3A3634" stroke-width="1"/>
                            <rect x="95" y="50" width="30" height="20" rx="3" fill="none" stroke="#3A3634" stroke-width="1"/>
                            <rect x="65" y="75" width="30" height="20" rx="3" fill="none" stroke="#3A3634" stroke-width="1"/>
                            <line x1="35" y1="20" x2="65" y2="20" stroke="#2A2624" stroke-width="0.5"/>
                            <line x1="95" y1="20" x2="125" y2="20" stroke="#2A2624" stroke-width="0.5"/>
                            <line x1="50" y1="30" x2="50" y2="50" stroke="#2A2624" stroke-width="0.5"/>
                            <line x1="110" y1="30" x2="110" y2="50" stroke="#2A2624" stroke-width="0.5"/>
                            <line x1="65" y1="60" x2="80" y2="75" stroke="#2A2624" stroke-width="0.5"/>
                            <line x1="95" y1="60" x2="80" y2="75" stroke="#2A2624" stroke-width="0.5"/>
                        </svg>
                    </div>
                    <div class="schematic-compact-label">
                        <span>System</span>
                        <span class="sc-dot"></span>
                    </div>
                </div>
            </div>

            <div class="nav-footer">
                <div class="nav-footer-status">
                    <span class="pulse-dot" id="runner-dot"></span>
                    <span id="runner-status-text">Connecting...</span>
                </div>
            </div>
        </nav>

        <!-- ═══ Main Content ═══ -->
        <div class="main-content">
            <div class="shell">
                <!-- Topbar -->
                <header class="topbar" role="banner">
                    <div class="topbar-brand">
                        <div class="topbar-title">
                            <h1 id="section-title">Overview</h1>
                            <span class="subtitle" id="section-subtitle">System status and quick actions</span>
                        </div>
                    </div>
                    <div class="topbar-actions">
                        <div class="topbar-status">
                            <span class="pulse-dot" id="topbar-dot"></span>
                            <span id="topbar-status-text">v2.4.0</span>
                        </div>
                        <button class="btn btn-outline" onclick="refreshAll()">Refresh</button>
                    </div>
                </header>

                <main id="main" role="main">

                <!-- ═══ SECTION: Overview ═══ -->
                <div class="dash-section active" id="sec-overview">

                <!-- Schematic Hero - Live System Architecture -->
                <div class="schematic-hero" id="system-schematic" data-live="true" role="img" aria-label="SLATE system architecture diagram showing connected services">
                    <div class="schematic-overlay">
                        <span class="schematic-title">System Architecture</span>
                        <span class="schematic-live-badge">
                            <span class="schematic-live-dot"></span>
                            Live
                        </span>
                    </div>
                    <div class="schematic-content" id="schematic-svg-container">
                        <div class="schematic-loading">
                            <div class="schematic-loading-spinner"></div>
                            <span>Loading schematic...</span>
                        </div>
                    </div>
                </div>

                <div class="grid">
                    <!-- Quick Actions -->
                    <div class="card col-12" role="region" aria-label="Quick Actions">
                        <div class="card-header">
                            <span class="card-title">Quick Actions</span>
                            <span style="font-size:0.6rem;color:var(--sl-text-disabled);">Common operations</span>
                        </div>
                        <div class="ctrl-grid" style="grid-template-columns: repeat(4, 1fr);">
                            <button class="ctrl-btn" onclick="slateCtrl('run-protocol', this)">
                                <span class="ctrl-dot" id="dot-protocol"></span>
                                <span class="ctrl-icon">&#9881;</span>
                                <span class="ctrl-label">Health Check</span>
                            </button>
                            <button class="ctrl-btn" onclick="slateCtrl('debug', this)">
                                <span class="ctrl-dot" id="dot-debug"></span>
                                <span class="ctrl-icon">&#128269;</span>
                                <span class="ctrl-label">Diagnostics</span>
                            </button>
                            <button class="ctrl-btn" onclick="slateDeploy('start', this)">
                                <span class="ctrl-dot" id="dot-deploy"></span>
                                <span class="ctrl-icon">&#9654;</span>
                                <span class="ctrl-label">Start Services</span>
                            </button>
                            <button class="ctrl-btn" onclick="dispatchWorkflow('ci.yml')">
                                <span class="ctrl-icon">&#9654;</span>
                                <span class="ctrl-label">Run CI</span>
                            </button>
                        </div>
                    </div>

                    <!-- System Health -->
                    <div class="card col-6">
                        <div class="card-header">
                            <span class="card-title">System Health</span>
                            <button class="card-action" onclick="refreshSystemHealth()">Refresh</button>
                        </div>
                        <div>
                            <div class="hw-title">Resources</div>
                            <div class="metric">
                                <span class="metric-label">CPU</span>
                                <div class="metric-bar"><div class="metric-fill" id="cpu-bar" style="width:0%"></div></div>
                                <span class="metric-value" id="cpu-value">--%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Memory</span>
                                <div class="metric-bar"><div class="metric-fill" id="memory-bar" style="width:0%"></div></div>
                                <span class="metric-value" id="memory-value">--%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Disk</span>
                                <div class="metric-bar"><div class="metric-fill" id="disk-bar" style="width:0%"></div></div>
                                <span class="metric-value" id="disk-value">--%</span>
                            </div>
                        </div>
                        <div style="margin-top: var(--sl-space-4);">
                            <div class="hw-title">GPU</div>
                            <div id="gpu-utilization"><div class="empty">Loading GPU...</div></div>
                        </div>
                    </div>

                    <!-- Services -->
                    <div class="card col-6">
                        <div class="card-header">
                            <span class="card-title">Services</span>
                            <button class="card-action" onclick="refreshServices()">Refresh</button>
                        </div>
                        <div id="services-list">
                            <div class="svc-item"><div class="svc-name"><div class="svc-icon">D</div><span>Dashboard</span></div><span class="badge online">Online</span></div>
                            <div class="svc-item"><div class="svc-name"><div class="svc-icon">O</div><span>Orchestrator</span></div><span class="badge pending" id="orch-badge">Checking</span></div>
                            <div class="svc-item"><div class="svc-name"><div class="svc-icon">R</div><span>Runner</span></div><span class="badge pending" id="runner-badge">Checking</span></div>
                            <div class="svc-item"><div class="svc-name"><div class="svc-icon">A</div><span>Ollama</span></div><span class="badge pending" id="ollama-badge">Checking</span></div>
                            <div class="svc-item"><div class="svc-name"><div class="svc-icon">F</div><span>Foundry</span></div><span class="badge pending" id="foundry-badge">Checking</span></div>
                            <div class="svc-item"><div class="svc-name"><div class="svc-icon">&#128051;</div><span>Docker</span></div><span class="badge pending" id="docker-badge">Checking</span></div>
                        </div>
                    </div>

                    <!-- Pipeline -->
                    <div class="card col-12" role="region" aria-label="Pipeline">
                        <div class="card-header">
                            <span class="card-title">Workflow Pipeline</span>
                            <button class="card-action" onclick="refreshWorkflowPipeline()">Refresh</button>
                        </div>
                        <div class="pipeline" id="flow-pipeline">
                            <div class="pipe-node">
                                <div class="pipe-icon" id="flow-task">T</div>
                                <div class="pipe-label">Tasks</div>
                                <div class="pipe-value" id="flow-task-count">0</div>
                            </div>
                            <div class="pipe-connector" id="flow-conn-1"></div>
                            <div class="pipe-node">
                                <div class="pipe-icon" id="flow-runner">R</div>
                                <div class="pipe-label">Runner</div>
                                <div class="pipe-value" id="flow-runner-status">Idle</div>
                            </div>
                            <div class="pipe-connector" id="flow-conn-2"></div>
                            <div class="pipe-node">
                                <div class="pipe-icon" id="flow-workflow">W</div>
                                <div class="pipe-label">Workflows</div>
                                <div class="pipe-value" id="flow-workflow-count">0</div>
                            </div>
                            <div class="pipe-connector" id="flow-conn-3"></div>
                            <div class="pipe-node">
                                <div class="pipe-icon" id="flow-pr">PR</div>
                                <div class="pipe-label">Results</div>
                                <div class="pipe-value" id="flow-pr-count">0</div>
                            </div>
                        </div>
                    </div>
                </div>
                </div>

                <!-- ═══ SECTION: Controls ═══ -->
                <div class="dash-section" id="sec-controls">
                <div class="grid">
                    <div class="card col-12" role="region" aria-label="SLATE Controls">
                        <div class="card-header">
                            <span class="card-title">System Controls</span>
                            <div style="display:flex;gap:var(--sl-space-2);align-items:center;">
                                <button class="card-action" onclick="loadAiRecommend()">AI Suggest</button>
                                <button class="card-action" onclick="toggleTerm()">Output</button>
                            </div>
                        </div>
                        <div id="ai-rec" style="display:none;padding:var(--sl-space-3);margin-bottom:var(--sl-space-3);background:var(--sl-accent-container);border:1px solid var(--sl-accent-dim);border-radius:var(--sl-radius-sm);font-size:0.75rem;color:var(--sl-accent-on-container);line-height:1.6;">
                            <span style="font-weight:700;margin-right:var(--sl-space-2);">&#9672; AI:</span>
                            <span id="ai-rec-text">Analyzing system state...</span>
                        </div>
                        <div class="ctrl-grid">
                            <button class="ctrl-btn" id="ctrl-protocol" onclick="slateCtrl('run-protocol', this)">
                                <span class="ctrl-dot" id="dot-protocol2"></span>
                                <span class="ctrl-icon">&#9881;</span>
                                <span class="ctrl-label">Run SLATE</span>
                            </button>
                            <button class="ctrl-btn" id="ctrl-update" onclick="slateCtrl('update', this)">
                                <span class="ctrl-dot" id="dot-update"></span>
                                <span class="ctrl-icon">&#8635;</span>
                                <span class="ctrl-label">Update</span>
                            </button>
                            <button class="ctrl-btn" id="ctrl-debug" onclick="slateCtrl('debug', this)">
                                <span class="ctrl-dot" id="dot-debug2"></span>
                                <span class="ctrl-icon">&#128269;</span>
                                <span class="ctrl-label">Debug</span>
                            </button>
                            <button class="ctrl-btn" id="ctrl-benchmark" onclick="slateCtrl('benchmark', this)">
                                <span class="ctrl-dot" id="dot-benchmark"></span>
                                <span class="ctrl-icon">&#9889;</span>
                                <span class="ctrl-label">Benchmark</span>
                            </button>
                            <button class="ctrl-btn" id="ctrl-deploy" onclick="slateDeploy('start', this)">
                                <span class="ctrl-dot" id="dot-deploy2"></span>
                                <span class="ctrl-icon">&#9654;</span>
                                <span class="ctrl-label">Deploy</span>
                            </button>
                            <button class="ctrl-btn" id="ctrl-security" onclick="slateCtrl('security', this)">
                                <span class="ctrl-dot" id="dot-security"></span>
                                <span class="ctrl-icon">&#128274;</span>
                                <span class="ctrl-label">Security</span>
                            </button>
                            <button class="ctrl-btn" id="ctrl-agents" onclick="slateCtrl('agents', this)">
                                <span class="ctrl-dot" id="dot-agents"></span>
                                <span class="ctrl-icon">&#129302;</span>
                                <span class="ctrl-label">Agents</span>
                            </button>
                            <button class="ctrl-btn" id="ctrl-gpu" onclick="slateCtrl('gpu', this)">
                                <span class="ctrl-dot" id="dot-gpu"></span>
                                <span class="ctrl-icon">&#9670;</span>
                                <span class="ctrl-label">GPU</span>
                            </button>
                        </div>
                        <div class="ctrl-term" id="ctrl-term"></div>
                        <div class="ci-row">
                            <button class="btn btn-filled" onclick="dispatchWorkflow('ci.yml')">CI Pipeline</button>
                            <button class="btn btn-outline" onclick="dispatchWorkflow('slate.yml')">SLATE Checks</button>
                            <button class="btn btn-outline" onclick="dispatchWorkflow('nightly.yml')">Nightly</button>
                            <button class="btn btn-outline" onclick="dispatchWorkflow('agentic.yml')">Agentic AI</button>
                        </div>
                    </div>

                    <!-- Action History / Tracking -->
                    <div class="card col-12" role="region" aria-label="Action Tracking">
                        <div class="card-header">
                            <span class="card-title">Action History</span>
                            <button class="card-action" onclick="refreshActionHistory()">Refresh</button>
                        </div>
                        <div id="action-history" style="max-height:240px;overflow-y:auto;">
                            <div class="empty">No actions recorded yet</div>
                        </div>
                    </div>
                </div>
                </div>

                <!-- ═══ SECTION: Hardware ═══ -->
                <div class="dash-section" id="sec-hardware">
                <div class="grid">
                    <div class="card col-6">
                        <div class="card-header">
                            <span class="card-title">GPU Performance</span>
                            <button class="card-action" onclick="runBenchmark()">Benchmark</button>
                        </div>
                        <div id="hw-gpu-meters">
                            <div class="metric"><span class="metric-label">GPU 0</span><div class="metric-bar"><div class="metric-fill" id="hw-gpu0-bar" style="width:0%"></div></div><span class="metric-value" id="hw-gpu0-val">--%</span></div>
                            <div class="metric"><span class="metric-label">GPU 1</span><div class="metric-bar"><div class="metric-fill" id="hw-gpu1-bar" style="width:0%"></div></div><span class="metric-value" id="hw-gpu1-val">--%</span></div>
                        </div>
                        <div style="display:flex;gap:var(--sl-space-4);margin-top:var(--sl-space-4);">
                            <div class="hw-stat"><div class="hw-stat-value" id="bench-speed">--</div><div class="hw-stat-label">tok/s</div></div>
                            <div class="hw-stat"><div class="hw-stat-value" id="bench-bandwidth">--</div><div class="hw-stat-label">GB/s</div></div>
                        </div>
                        <!-- GPU Topology Card Schematic (Spec 012 Phase 2) -->
                        <div class="card-schematic" id="gpu-topology-schematic" onclick="openSchematicModal('inference')" title="GPU topology — click to expand">
                            <button class="cs-expand" aria-label="Expand schematic">&#x26F6;</button>
                            <svg viewBox="0 0 300 80" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
                                <rect x="5" y="10" width="60" height="28" rx="4" fill="none" stroke="#3A3634" stroke-width="1"/>
                                <text x="35" y="27" text-anchor="middle" fill="#666" font-size="8" font-family="Consolas, monospace">GPU 0</text>
                                <rect x="120" y="10" width="60" height="28" rx="4" fill="none" stroke="#3A3634" stroke-width="1"/>
                                <text x="150" y="27" text-anchor="middle" fill="#666" font-size="8" font-family="Consolas, monospace">GPU 1</text>
                                <rect x="235" y="10" width="60" height="28" rx="4" fill="none" stroke="#3A3634" stroke-width="1"/>
                                <text x="265" y="27" text-anchor="middle" fill="#666" font-size="8" font-family="Consolas, monospace">Ollama</text>
                                <rect x="5" y="50" width="60" height="22" rx="4" fill="none" stroke="#2A2624" stroke-width="0.8"/>
                                <text x="35" y="64" text-anchor="middle" fill="#555" font-size="7" font-family="Consolas, monospace">16GB VRAM</text>
                                <rect x="120" y="50" width="60" height="22" rx="4" fill="none" stroke="#2A2624" stroke-width="0.8"/>
                                <text x="150" y="64" text-anchor="middle" fill="#555" font-size="7" font-family="Consolas, monospace">16GB VRAM</text>
                                <line x1="65" y1="24" x2="120" y2="24" stroke="#2A2624" stroke-width="0.8" stroke-dasharray="3,2"/>
                                <line x1="180" y1="24" x2="235" y2="24" stroke="#2A2624" stroke-width="0.8" stroke-dasharray="3,2"/>
                                <line x1="35" y1="38" x2="35" y2="50" stroke="#2A2624" stroke-width="0.5"/>
                                <line x1="150" y1="38" x2="150" y2="50" stroke="#2A2624" stroke-width="0.5"/>
                            </svg>
                        </div>
                    </div>

                    <div class="card col-6">
                        <div class="card-header">
                            <span class="card-title">Ollama Models</span>
                            <button class="card-action" onclick="refreshSystemHealth()">Refresh</button>
                        </div>
                        <div id="ollama-status"><div class="empty">Loading...</div></div>
                    </div>

                    <div class="card col-6">
                        <div class="card-header">
                            <span class="card-title">Docker</span>
                            <button class="card-action" onclick="refreshDocker()">Refresh</button>
                        </div>
                        <div class="docker-grid" id="docker-containers"><div class="empty">Loading...</div></div>
                        <div style="display:flex;gap:var(--sl-space-2);margin-top:var(--sl-space-3);padding-top:var(--sl-space-3);border-top:1px solid var(--sl-border);">
                            <button class="btn btn-outline" onclick="dockerActionAll('start')" style="flex:1;font-size:0.65rem;">Start All</button>
                            <button class="btn btn-outline" onclick="dockerActionAll('stop')" style="flex:1;font-size:0.65rem;">Stop All</button>
                        </div>
                    </div>

                    <div class="card col-6">
                        <div class="card-header">
                            <span class="card-title">Multi-Runner</span>
                            <button class="card-action" onclick="refreshMultiRunner()">Refresh</button>
                        </div>
                        <div id="multi-runner-status"><div class="empty">Loading...</div></div>
                    </div>
                </div>
                </div>

                <!-- ═══ SECTION: Workflows ═══ -->
                <div class="dash-section" id="sec-workflows">
                <div class="grid">
                    <div class="card col-8">
                        <div class="card-header">
                            <span class="card-title">Recent Workflows</span>
                            <button class="card-action" onclick="refreshWorkflows()">Refresh</button>
                        </div>
                        <div id="workflow-list" style="max-height:400px;overflow-y:auto;"><div class="empty">Loading...</div></div>
                    </div>

                    <div class="card col-4">
                        <div class="card-header">
                            <span class="card-title">Tech Tree</span>
                            <button class="card-action" onclick="refreshTechTree()">Refresh</button>
                        </div>
                        <div class="tech-tree-stats" id="tech-tree-stats"><div class="empty">Loading...</div></div>
                    </div>

                    <div class="card col-6">
                        <div class="card-header">
                            <span class="card-title">Specifications</span>
                            <button class="card-action" onclick="refreshSpecs()">Refresh</button>
                        </div>
                        <div id="spec-list"><div class="empty">Loading...</div></div>
                    </div>

                    <div class="card col-6">
                        <div class="card-header">
                            <span class="card-title">Forks</span>
                            <button class="card-action" onclick="refreshForks()">Refresh</button>
                        </div>
                        <div id="fork-list"><div class="empty">Loading...</div></div>
                    </div>
                </div>
                </div>

                <!-- ═══ SECTION: Tasks ═══ -->
                <div class="dash-section" id="sec-tasks">
                <div class="grid">
                    <div class="card col-12">
                        <div class="card-header">
                            <span class="card-title">Task Queue</span>
                            <button class="card-action" onclick="refreshTasks()">Refresh</button>
                        </div>
                        <div class="task-form">
                            <input type="text" class="task-input" id="new-task-title" placeholder="Add a new task..." onkeypress="if(event.key==='Enter')createTask()">
                            <select id="new-task-priority">
                                <option value="3">Normal</option>
                                <option value="1">High</option>
                                <option value="5">Low</option>
                            </select>
                            <button class="btn btn-tonal" onclick="createTask()">Add</button>
                        </div>
                        <div id="task-list" style="max-height:400px;overflow-y:auto;"><div class="empty">Loading tasks...</div></div>
                    </div>

                    <div class="card col-12">
                        <div class="card-header">
                            <span class="card-title">Task Activity Heatmap</span>
                            <button class="card-action" onclick="refreshHeatmap()">Refresh</button>
                        </div>
                        <div id="heatmap-total" style="font-size:0.65rem;color:var(--sl-text-disabled);margin-bottom:var(--sl-space-2);">0 tasks</div>
                        <div class="heatmap-grid" id="heatmap-grid"></div>
                    </div>
                </div>
                </div>

                <!-- ═══ SECTION: GitHub ═══ -->
                <div class="dash-section" id="sec-github">
                <div class="grid">
                    <div class="card col-12">
                        <div class="card-header">
                            <span class="card-title">GitHub Integration</span>
                            <button class="card-action" onclick="refreshGitHub()">Refresh</button>
                        </div>
                        <div class="gh-grid">
                            <div>
                                <div class="gh-section-title">Open PRs <span class="count" id="pr-count">0</span></div>
                                <div id="pr-list"><div class="empty">Loading...</div></div>
                            </div>
                            <div>
                                <div class="gh-section-title">Commits</div>
                                <div id="commit-list"><div class="empty">Loading...</div></div>
                            </div>
                            <div>
                                <div class="gh-section-title">Issues <span class="count" id="issue-count">0</span></div>
                                <div id="issue-list"><div class="empty">Loading...</div></div>
                            </div>
                            <div>
                                <div class="gh-section-title">Release</div>
                                <div id="release-info"><div class="empty">Loading...</div></div>
                            </div>
                        </div>
                    </div>
                </div>
                </div>

                <!-- ═══ SECTION: Agents ═══ -->
                <div class="dash-section" id="sec-agents">
                <div class="grid">
                    <div class="card col-12">
                        <div class="card-header">
                            <span class="card-title">Agent System</span>
                            <button class="card-action" onclick="refreshAgents()">Refresh</button>
                        </div>
                        <!-- Agent Pipeline Card Schematic (Spec 012 Phase 2) -->
                        <div class="card-schematic" id="agent-pipeline-schematic" onclick="openSchematicModal('system')" title="Agent pipeline — click to expand">
                            <button class="cs-expand" aria-label="Expand schematic">&#x26F6;</button>
                            <svg viewBox="0 0 400 60" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
                                <rect x="5" y="15" width="50" height="30" rx="4" fill="none" stroke="#3A3634" stroke-width="1"/>
                                <text x="30" y="34" text-anchor="middle" fill="#666" font-size="7" font-family="Consolas, monospace">ALPHA</text>
                                <rect x="75" y="15" width="50" height="30" rx="4" fill="none" stroke="#3A3634" stroke-width="1"/>
                                <text x="100" y="34" text-anchor="middle" fill="#666" font-size="7" font-family="Consolas, monospace">BETA</text>
                                <rect x="145" y="15" width="50" height="30" rx="4" fill="none" stroke="#3A3634" stroke-width="1"/>
                                <text x="170" y="34" text-anchor="middle" fill="#666" font-size="7" font-family="Consolas, monospace">GAMMA</text>
                                <rect x="215" y="15" width="50" height="30" rx="4" fill="none" stroke="#3A3634" stroke-width="1"/>
                                <text x="240" y="34" text-anchor="middle" fill="#666" font-size="7" font-family="Consolas, monospace">DELTA</text>
                                <rect x="285" y="15" width="56" height="30" rx="4" fill="none" stroke="#3A3634" stroke-width="1"/>
                                <text x="313" y="34" text-anchor="middle" fill="#666" font-size="7" font-family="Consolas, monospace">COPILOT</text>
                                <line x1="55" y1="30" x2="75" y2="30" stroke="#2A2624" stroke-width="0.8" stroke-dasharray="2,2"/>
                                <line x1="125" y1="30" x2="145" y2="30" stroke="#2A2624" stroke-width="0.8" stroke-dasharray="2,2"/>
                                <line x1="195" y1="30" x2="215" y2="30" stroke="#2A2624" stroke-width="0.8" stroke-dasharray="2,2"/>
                                <line x1="265" y1="30" x2="285" y2="30" stroke="#2A2624" stroke-width="0.8" stroke-dasharray="2,2"/>
                                <rect x="360" y="18" width="35" height="24" rx="3" fill="none" stroke="#B85A3C" stroke-width="1" opacity="0.6"/>
                                <text x="378" y="34" text-anchor="middle" fill="#B85A3C" font-size="6" font-family="Consolas, monospace">ROUTER</text>
                                <line x1="341" y1="30" x2="360" y2="30" stroke="#B85A3C" stroke-width="0.6" opacity="0.4"/>
                            </svg>
                            <div class="schematic-status-overlay">
                                <span class="schematic-status-chip"><span class="ssc-dot active"></span>ALPHA</span>
                                <span class="schematic-status-chip"><span class="ssc-dot active"></span>BETA</span>
                                <span class="schematic-status-chip"><span class="ssc-dot active"></span>GAMMA</span>
                                <span class="schematic-status-chip"><span class="ssc-dot idle"></span>DELTA</span>
                                <span class="schematic-status-chip"><span class="ssc-dot active"></span>COPILOT</span>
                            </div>
                        </div>
                        <div id="agent-list"><div class="empty">Loading agents...</div></div>
                    </div>
                </div>
                </div>

                <!-- ═══ SECTION: Activity ═══ -->
                <div class="dash-section" id="sec-activity">
                <div class="grid">
                    <div class="card col-12">
                        <div class="card-header">
                            <span class="card-title">Activity Feed</span>
                            <button class="card-action" onclick="refreshActivity()">Refresh</button>
                        </div>
                        <div id="activity-feed" style="max-height:400px;overflow-y:auto;"><div class="empty">No activity</div></div>
                    </div>
                </div>
                </div>

                <!-- ═══ SECTION: Command Center (Watchmaker Phase 3) ═══ -->
                <div class="dash-section" id="sec-command-center">
                    <div class="dashboard-3d">
                        <!-- Health Ring Schematic -->
                        <div class="watchmaker-card active" style="margin-bottom:24px;">
                            <div class="watchmaker-card-header">
                                <svg class="gear-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="2"/>
                                    <path d="M12 1v3m0 16v3M1 12h3m16 0h3M4.22 4.22l2.12 2.12m11.3 11.3l2.12 2.12M4.22 19.78l2.12-2.12m11.3-11.3l2.12-2.12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                </svg>
                                <span style="font-weight:600;font-size:0.85rem;">System Health Ring</span>
                                <span style="margin-left:auto;font-size:0.6rem;color:var(--sl-text-disabled);">Real-time component status</span>
                            </div>
                            <div class="health-ring" id="health-ring">
                                <div class="health-ring-node" data-component="gpu">
                                    <span class="hr-icon">&#9883;</span>
                                    <span class="hr-label">GPU</span>
                                    <span class="hr-value" id="hr-gpu-val">--</span>
                                    <span class="status-jewel active" id="hr-gpu-jewel"></span>
                                </div>
                                <div class="health-ring-connector"></div>
                                <div class="health-ring-node" data-component="ai">
                                    <span class="hr-icon">&#129302;</span>
                                    <span class="hr-label">AI</span>
                                    <span class="hr-value" id="hr-ai-val">--</span>
                                    <span class="status-jewel active" id="hr-ai-jewel"></span>
                                </div>
                                <div class="health-ring-connector"></div>
                                <div class="health-ring-node" data-component="tasks">
                                    <span class="hr-icon">&#9744;</span>
                                    <span class="hr-label">Tasks</span>
                                    <span class="hr-value" id="hr-tasks-val">--</span>
                                    <span class="status-jewel pending" id="hr-tasks-jewel"></span>
                                </div>
                                <div class="health-ring-connector"></div>
                                <div class="health-ring-node" data-component="runner">
                                    <span class="hr-icon">&#9654;</span>
                                    <span class="hr-label">Runner</span>
                                    <span class="hr-value" id="hr-runner-val">--</span>
                                    <span class="status-jewel pending" id="hr-runner-jewel"></span>
                                </div>
                                <div class="health-ring-connector"></div>
                                <div class="health-ring-node" data-component="services">
                                    <span class="hr-icon">&#9670;</span>
                                    <span class="hr-label">Services</span>
                                    <span class="hr-value" id="hr-svc-val">--</span>
                                    <span class="status-jewel active" id="hr-svc-jewel"></span>
                                </div>
                            </div>
                        </div>

                        <!-- Summary Cards Grid -->
                        <div class="grid">
                            <div class="col-3">
                                <div class="watchmaker-card active" onclick="showSection('constellation',null)">
                                    <div class="watchmaker-card-header">
                                        <span class="status-jewel active" id="cc-svc-jewel"></span>
                                        <span style="font-size:0.7rem;font-weight:600;">Services</span>
                                    </div>
                                    <div style="font-size:1.6rem;font-weight:700;font-family:var(--sl-font-mono);" id="cc-svc-count">--</div>
                                    <div style="font-size:0.55rem;color:var(--sl-text-disabled);margin-top:4px;">Active connections</div>
                                </div>
                            </div>
                            <div class="col-3">
                                <div class="watchmaker-card active" onclick="showSection('gpu-workbench',null)">
                                    <div class="watchmaker-card-header">
                                        <span class="status-jewel active" id="cc-gpu-jewel"></span>
                                        <span style="font-size:0.7rem;font-weight:600;">GPU Farm</span>
                                    </div>
                                    <div style="font-size:1.6rem;font-weight:700;font-family:var(--sl-font-mono);" id="cc-gpu-count">2x</div>
                                    <div style="font-size:0.55rem;color:var(--sl-text-disabled);margin-top:4px;">RTX 5070 Ti &middot; 32GB</div>
                                </div>
                            </div>
                            <div class="col-3">
                                <div class="watchmaker-card" onclick="showSection('task-orch',null)">
                                    <div class="watchmaker-card-header">
                                        <span class="status-jewel pending" id="cc-task-jewel"></span>
                                        <span style="font-size:0.7rem;font-weight:600;">Task Queue</span>
                                    </div>
                                    <div style="font-size:1.6rem;font-weight:700;font-family:var(--sl-font-mono);" id="cc-task-count">--</div>
                                    <div style="font-size:0.55rem;color:var(--sl-text-disabled);margin-top:4px;">Pending tasks</div>
                                </div>
                            </div>
                            <div class="col-3">
                                <div class="watchmaker-card active">
                                    <div class="watchmaker-card-header">
                                        <span class="status-jewel active" id="cc-model-jewel"></span>
                                        <span style="font-size:0.7rem;font-weight:600;">AI Models</span>
                                    </div>
                                    <div style="font-size:1.6rem;font-weight:700;font-family:var(--sl-font-mono);" id="cc-model-count">--</div>
                                    <div style="font-size:0.55rem;color:var(--sl-text-disabled);margin-top:4px;">Loaded in Ollama</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- ═══ SECTION: Service Constellation (Watchmaker Phase 3) ═══ -->
                <div class="dash-section" id="sec-constellation">
                    <div class="dashboard-3d">
                        <div class="watchmaker-card active" style="padding:0;overflow:hidden;">
                            <div style="padding:20px 24px;border-bottom:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;gap:12px;">
                                <svg class="gear-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="2"/>
                                    <path d="M12 1v3m0 16v3M1 12h3m16 0h3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                </svg>
                                <span style="font-weight:600;font-size:0.85rem;">Service Constellation</span>
                                <span style="font-size:0.6rem;color:var(--sl-text-disabled);margin-left:auto;" id="constellation-count">Loading...</span>
                            </div>
                            <div class="constellation-view" id="constellation-svg">
                                <svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:400px;">
                                    <!-- Dashboard node (center top) -->
                                    <rect x="345" y="30" width="110" height="40" rx="8" fill="rgba(184,90,60,0.1)" stroke="#B85A3C" stroke-width="1.5"/>
                                    <text x="400" y="55" text-anchor="middle" fill="#B85A3C" font-size="12" font-family="Consolas,monospace" font-weight="600">Dashboard</text>
                                    <circle cx="360" cy="45" r="4" id="cst-dash-jewel" fill="#22C55E"/>
                                    <text x="400" y="25" text-anchor="middle" fill="#666" font-size="9" font-family="Consolas,monospace">:8080</text>

                                    <!-- Connection lines from Dashboard -->
                                    <line x1="370" y1="70" x2="180" y2="130" class="data-flow-line" stroke-width="1.5"/>
                                    <line x1="400" y1="70" x2="400" y2="130" class="data-flow-line" stroke-width="1.5"/>
                                    <line x1="430" y1="70" x2="620" y2="130" class="data-flow-line" stroke-width="1.5"/>

                                    <!-- Ollama node (left) -->
                                    <rect x="125" y="130" width="110" height="40" rx="8" fill="rgba(34,197,94,0.08)" stroke="#22C55E" stroke-width="1"/>
                                    <text x="180" y="155" text-anchor="middle" fill="#98C1D9" font-size="12" font-family="Consolas,monospace">Ollama</text>
                                    <circle cx="140" cy="145" r="4" id="cst-ollama-jewel" fill="#22C55E"/>
                                    <text x="180" y="125" text-anchor="middle" fill="#666" font-size="9" font-family="Consolas,monospace">:11434</text>

                                    <!-- ChromaDB node (center) -->
                                    <rect x="345" y="130" width="110" height="40" rx="8" fill="rgba(34,197,94,0.08)" stroke="#22C55E" stroke-width="1"/>
                                    <text x="400" y="155" text-anchor="middle" fill="#98C1D9" font-size="12" font-family="Consolas,monospace">ChromaDB</text>
                                    <circle cx="360" cy="145" r="4" id="cst-chroma-jewel" fill="#22C55E"/>
                                    <text x="400" y="125" text-anchor="middle" fill="#666" font-size="9" font-family="Consolas,monospace">:8000</text>

                                    <!-- MCP Server node (right) -->
                                    <rect x="565" y="130" width="110" height="40" rx="8" fill="rgba(34,197,94,0.08)" stroke="#22C55E" stroke-width="1"/>
                                    <text x="620" y="155" text-anchor="middle" fill="#98C1D9" font-size="12" font-family="Consolas,monospace">MCP Server</text>
                                    <circle cx="580" cy="145" r="4" id="cst-mcp-jewel" fill="#6B7280"/>
                                    <text x="620" y="125" text-anchor="middle" fill="#666" font-size="9" font-family="Consolas,monospace">:stdio</text>

                                    <!-- Lower connections -->
                                    <line x1="180" y1="170" x2="180" y2="240" class="data-flow-line" stroke-width="1"/>
                                    <line x1="400" y1="170" x2="290" y2="240" class="data-flow-line" stroke-width="1"/>
                                    <line x1="400" y1="170" x2="510" y2="240" class="data-flow-line" stroke-width="1"/>

                                    <!-- Foundry node -->
                                    <rect x="125" y="240" width="110" height="40" rx="8" fill="rgba(152,193,217,0.08)" stroke="#98C1D9" stroke-width="1"/>
                                    <text x="180" y="265" text-anchor="middle" fill="#98C1D9" font-size="12" font-family="Consolas,monospace">Foundry</text>
                                    <circle cx="140" cy="255" r="4" id="cst-foundry-jewel" fill="#6B7280"/>
                                    <text x="180" y="235" text-anchor="middle" fill="#666" font-size="9" font-family="Consolas,monospace">:5272</text>

                                    <!-- GPU Farm node -->
                                    <rect x="235" y="240" width="110" height="40" rx="8" fill="rgba(184,90,60,0.1)" stroke="#B85A3C" stroke-width="1.5"/>
                                    <text x="290" y="265" text-anchor="middle" fill="#B85A3C" font-size="12" font-family="Consolas,monospace" font-weight="600">GPU Farm</text>
                                    <circle cx="250" cy="255" r="4" id="cst-gpu-jewel" fill="#22C55E"/>
                                    <text x="290" y="235" text-anchor="middle" fill="#666" font-size="9" font-family="Consolas,monospace">2x RTX 5070 Ti</text>

                                    <!-- Runner node -->
                                    <rect x="455" y="240" width="110" height="40" rx="8" fill="rgba(152,193,217,0.08)" stroke="#98C1D9" stroke-width="1"/>
                                    <text x="510" y="265" text-anchor="middle" fill="#98C1D9" font-size="12" font-family="Consolas,monospace">Runner</text>
                                    <circle cx="470" cy="255" r="4" id="cst-runner-jewel" fill="#F59E0B"/>
                                    <text x="510" y="235" text-anchor="middle" fill="#666" font-size="9" font-family="Consolas,monospace">Actions</text>

                                    <!-- GPU interconnect -->
                                    <line x1="290" y1="280" x2="290" y2="340" stroke="#B85A3C" stroke-width="1" stroke-dasharray="4,3"/>
                                    <rect x="235" y="340" width="110" height="30" rx="6" fill="rgba(184,90,60,0.06)" stroke="#B85A3C" stroke-width="0.8"/>
                                    <text x="290" y="360" text-anchor="middle" fill="#666" font-size="9" font-family="Consolas,monospace">32GB VRAM Total</text>
                                </svg>
                            </div>
                            <div class="constellation-legend">
                                <div class="constellation-legend-item"><span class="status-jewel active" style="width:8px;height:8px;"></span> Active</div>
                                <div class="constellation-legend-item"><span class="status-jewel pending" style="width:8px;height:8px;"></span> Standby</div>
                                <div class="constellation-legend-item"><span class="status-jewel inactive" style="width:8px;height:8px;"></span> Offline</div>
                                <div class="constellation-legend-item" style="margin-left:auto;">&#8212; Data Flow</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- ═══ SECTION: GPU Workbench (Watchmaker Phase 3) ═══ -->
                <div class="dash-section" id="sec-gpu-workbench">
                    <div class="dashboard-3d">
                        <div class="gpu-workbench">
                            <div class="gpu-unit" id="gpu-unit-0">
                                <div class="gpu-unit-header">
                                    <span class="gpu-unit-title">GPU 0: RTX 5070 Ti</span>
                                    <span class="status-jewel active" id="wb-gpu0-jewel"></span>
                                </div>
                                <div class="gpu-meters">
                                    <div class="gpu-meter-row">
                                        <span class="gpu-meter-label">&#9883; Compute</span>
                                        <div class="gpu-meter-bar"><div class="gpu-meter-fill" id="wb-gpu0-compute" style="width:0%"></div></div>
                                        <span class="gpu-meter-value" id="wb-gpu0-compute-val">--%</span>
                                    </div>
                                    <div class="gpu-meter-row">
                                        <span class="gpu-meter-label">&#9673; Memory</span>
                                        <div class="gpu-meter-bar"><div class="gpu-meter-fill" id="wb-gpu0-memory" style="width:0%"></div></div>
                                        <span class="gpu-meter-value" id="wb-gpu0-memory-val">--%</span>
                                    </div>
                                    <div class="gpu-meter-row">
                                        <span class="gpu-meter-label">&#9889; Power</span>
                                        <div class="gpu-meter-bar"><div class="gpu-meter-fill" id="wb-gpu0-power" style="width:0%"></div></div>
                                        <span class="gpu-meter-value" id="wb-gpu0-power-val">--W</span>
                                    </div>
                                </div>
                                <div class="gpu-tasks-list" id="wb-gpu0-tasks">
                                    <div class="gpu-task-item"><span class="status-jewel active" style="width:6px;height:6px;"></span> Loading...</div>
                                </div>
                                <div style="margin-top:12px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.04);font-size:0.55rem;color:var(--sl-text-disabled);font-family:var(--sl-font-mono);">
                                    <span id="wb-gpu0-temp">Temp: --&deg;C</span> &middot; <span>CUDA 12.8</span> &middot; <span>Blackwell</span>
                                </div>
                            </div>

                            <div class="gpu-unit" id="gpu-unit-1">
                                <div class="gpu-unit-header">
                                    <span class="gpu-unit-title">GPU 1: RTX 5070 Ti</span>
                                    <span class="status-jewel active" id="wb-gpu1-jewel"></span>
                                </div>
                                <div class="gpu-meters">
                                    <div class="gpu-meter-row">
                                        <span class="gpu-meter-label">&#9883; Compute</span>
                                        <div class="gpu-meter-bar"><div class="gpu-meter-fill" id="wb-gpu1-compute" style="width:0%"></div></div>
                                        <span class="gpu-meter-value" id="wb-gpu1-compute-val">--%</span>
                                    </div>
                                    <div class="gpu-meter-row">
                                        <span class="gpu-meter-label">&#9673; Memory</span>
                                        <div class="gpu-meter-bar"><div class="gpu-meter-fill" id="wb-gpu1-memory" style="width:0%"></div></div>
                                        <span class="gpu-meter-value" id="wb-gpu1-memory-val">--%</span>
                                    </div>
                                    <div class="gpu-meter-row">
                                        <span class="gpu-meter-label">&#9889; Power</span>
                                        <div class="gpu-meter-bar"><div class="gpu-meter-fill" id="wb-gpu1-power" style="width:0%"></div></div>
                                        <span class="gpu-meter-value" id="wb-gpu1-power-val">--W</span>
                                    </div>
                                </div>
                                <div class="gpu-tasks-list" id="wb-gpu1-tasks">
                                    <div class="gpu-task-item"><span class="status-jewel active" style="width:6px;height:6px;"></span> Loading...</div>
                                </div>
                                <div style="margin-top:12px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.04);font-size:0.55rem;color:var(--sl-text-disabled);font-family:var(--sl-font-mono);">
                                    <span id="wb-gpu1-temp">Temp: --&deg;C</span> &middot; <span>CUDA 12.8</span> &middot; <span>Blackwell</span>
                                </div>
                            </div>

                            <div class="gpu-load-scheduler">
                                <span style="font-weight:600;font-size:0.7rem;white-space:nowrap;">GPU 0</span>
                                <div class="gpu-load-bar">
                                    <div class="gpu-load-fill-0" id="wb-load-0" style="width:50%"></div>
                                    <div class="gpu-load-fill-1" id="wb-load-1" style="width:30%"></div>
                                </div>
                                <span style="font-weight:600;font-size:0.7rem;white-space:nowrap;">GPU 1</span>
                                <span style="font-size:0.55rem;color:var(--sl-text-disabled);margin-left:8px;">Load Distribution</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- ═══ SECTION: Task Orchestration (Watchmaker Phase 3) ═══ -->
                <div class="dash-section" id="sec-task-orch">
                    <div class="dashboard-3d">
                        <div class="watchmaker-card active" style="margin-bottom:24px;">
                            <div class="watchmaker-card-header">
                                <svg class="gear-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="2"/>
                                    <path d="M12 1v3m0 16v3M1 12h3m16 0h3M4.22 4.22l2.12 2.12m11.3 11.3l2.12 2.12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                                </svg>
                                <span style="font-weight:600;font-size:0.85rem;">Task Orchestration Pipeline</span>
                                <button class="card-action" onclick="refreshTaskOrchestration()" style="margin-left:auto;">Refresh</button>
                            </div>
                            <div class="task-pipeline-viz" id="task-pipeline">
                                <div class="task-pipeline-stage" id="tp-pending">
                                    <div class="task-pipeline-stage-name">Pending</div>
                                    <div class="task-pipeline-stage-count" id="tp-pending-count">0</div>
                                    <span class="status-jewel pending" style="margin-top:8px;"></span>
                                </div>
                                <div class="task-pipeline-arrow">&raquo;</div>
                                <div class="task-pipeline-stage current" id="tp-running">
                                    <div class="task-pipeline-stage-name">Running</div>
                                    <div class="task-pipeline-stage-count" id="tp-running-count">0</div>
                                    <span class="status-jewel active" style="margin-top:8px;"></span>
                                </div>
                                <div class="task-pipeline-arrow">&raquo;</div>
                                <div class="task-pipeline-stage" id="tp-review">
                                    <div class="task-pipeline-stage-name">Review</div>
                                    <div class="task-pipeline-stage-count" id="tp-review-count">0</div>
                                    <span class="status-jewel pending" style="margin-top:8px;"></span>
                                </div>
                                <div class="task-pipeline-arrow">&raquo;</div>
                                <div class="task-pipeline-stage" id="tp-completed">
                                    <div class="task-pipeline-stage-name">Complete</div>
                                    <div class="task-pipeline-stage-count" id="tp-completed-count">0</div>
                                    <span class="status-jewel active" style="margin-top:8px;"></span>
                                </div>
                            </div>
                        </div>

                        <!-- Task List -->
                        <div class="grid">
                            <div class="card col-12">
                                <div class="card-header">
                                    <span class="card-title">Active Workflow Runs</span>
                                    <button class="card-action" onclick="refreshWorkflowRuns()">Refresh</button>
                                </div>
                                <div id="task-orch-runs"><div class="empty">Loading workflow runs...</div></div>
                            </div>
                        </div>
                    </div>
                </div>

                </main>
            </div>
        </div>
    </div>

    <div id="conn-status"><span class="pulse-dot" id="ws-dot"></span> <span id="ws-text">Connecting...</span></div>

    <!-- Schematic Tooltip (Spec 012 Phase 3) -->
    <div class="schematic-tooltip" id="schematic-tooltip">
        <div class="schematic-tooltip-title" id="tooltip-title">Component</div>
        <div class="schematic-tooltip-desc" id="tooltip-desc">Description</div>
        <div class="schematic-tooltip-status">
            <span class="sts-dot" id="tooltip-status-dot" style="background:#22C55E"></span>
            <span id="tooltip-status-text">Active</span>
        </div>
    </div>

    <!-- Schematic Modal Detail View (Spec 012 Phase 2) -->
    <div class="schematic-modal-overlay" id="schematic-modal" role="dialog" aria-modal="true" aria-label="Schematic detail view">
        <div class="schematic-modal">
            <div class="schematic-modal-header">
                <span class="schematic-modal-title" id="schematic-modal-title">System Architecture</span>
                <div class="schematic-modal-actions">
                    <div class="schematic-zoom-controls" style="position:static;margin-right:8px;">
                        <button class="schematic-zoom-btn" onclick="zoomSchematic(-zoomStep)" title="Zoom out">-</button>
                        <button class="schematic-zoom-btn" onclick="resetZoom()" title="Reset zoom">1:1</button>
                        <button class="schematic-zoom-btn" onclick="zoomSchematic(zoomStep)" title="Zoom in">+</button>
                    </div>
                    <span class="schematic-live-badge">
                        <span class="schematic-live-dot"></span>
                        Live
                    </span>
                    <button class="schematic-modal-close" onclick="closeSchematicModal()" aria-label="Close">&times;</button>
                </div>
            </div>
            <div class="schematic-modal-body" id="schematic-modal-body">
                <div class="schematic-loading">
                    <div class="schematic-loading-spinner"></div>
                    <span>Loading schematic...</span>
                </div>
            </div>
            <div class="schematic-modal-footer">
                <span id="schematic-modal-timestamp">Last updated: --</span>
                <span>Spec 012 &middot; Schematic SDK v1.1.0</span>
            </div>
        </div>
    </div>
'''


def build_template_js() -> str:
    """Build the JavaScript for the dashboard template."""
    # This JS preserves all the existing API-calling functions
    # from the original dashboard, adapted to the new element IDs
    return '''
    <script>
        // ═══════════════════════════════════════════════════════════════
        // SLATE Dashboard v3.0 — JavaScript
        // ═══════════════════════════════════════════════════════════════

        // ─── Onboarding ──────────────────────────────────────────────
        (function() {
            if (!localStorage.getItem('slate_onboarded')) {
                document.getElementById('onboard').classList.add('active');
            }
        })();

        function dismissOnboard() {
            localStorage.setItem('slate_onboarded', '1');
            document.getElementById('onboard').classList.remove('active');
        }

        // ─── Section Navigation ──────────────────────────────────────
        const sectionTitles = {
            overview:  { title: 'Overview',   sub: 'System status and quick actions' },
            controls:  { title: 'Controls',   sub: 'Manage system operations, CI/CD, and AI' },
            hardware:  { title: 'Hardware',   sub: 'GPU, Docker, and resource monitoring' },
            workflows: { title: 'Workflows',  sub: 'CI/CD pipelines, specs, and tech tree' },
            tasks:     { title: 'Tasks',      sub: 'Task queue, heatmap, and scheduling' },
            github:    { title: 'GitHub',     sub: 'PRs, issues, commits, and releases' },
            agents:    { title: 'Agents',     sub: 'AI agent system and routing' },
            activity:  { title: 'Activity',   sub: 'Recent operations and events' },
            'command-center': { title: 'Command Center', sub: 'Watchmaker system health overview' },
            'constellation':  { title: 'Service Constellation', sub: 'Interactive service topology map' },
            'gpu-workbench':  { title: 'GPU Workbench', sub: 'Dual RTX 5070 Ti workload management' },
            'task-orch':      { title: 'Task Orchestration', sub: 'Pipeline visualization and workflow runs' }
        };

        let activeSection = localStorage.getItem('slate_section') || 'overview';

        function showSection(name, navBtn) {
            // Hide all sections
            document.querySelectorAll('.dash-section').forEach(s => s.classList.remove('active'));
            // Show target
            const sec = document.getElementById('sec-' + name);
            if (sec) sec.classList.add('active');
            // Update nav active state
            document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
            if (navBtn) navBtn.classList.add('active');
            // Update topbar
            const info = sectionTitles[name] || { title: name, sub: '' };
            document.getElementById('section-title').textContent = info.title;
            document.getElementById('section-subtitle').textContent = info.sub;
            // Persist
            activeSection = name;
            localStorage.setItem('slate_section', name);
        }

        // Restore last section on load
        (function() {
            const saved = localStorage.getItem('slate_section') || 'overview';
            const navItems = document.querySelectorAll('.nav-item');
            const sectionNames = ['overview','controls','hardware','workflows','tasks','github','agents','activity','command-center','constellation','gpu-workbench','task-orch'];
            const idx = sectionNames.indexOf(saved);
            if (idx >= 0 && navItems[idx]) {
                showSection(saved, navItems[idx]);
            }
        })();

        // Action history tracking
        const actionLog = [];
        function logAction(action, success, duration) {
            actionLog.unshift({ action, success, duration, time: new Date().toLocaleTimeString() });
            if (actionLog.length > 50) actionLog.pop();
            refreshActionHistory();
        }
        function refreshActionHistory() {
            const el = document.getElementById('action-history');
            if (!el) return;
            if (actionLog.length === 0) { el.innerHTML = '<div class="empty">No actions recorded yet</div>'; return; }
            el.innerHTML = actionLog.map(a => {
                const icon = a.success ? '<span class="ok">&#10003;</span>' : '<span class="err">&#10007;</span>';
                const dur = a.duration ? ' (' + a.duration + 'ms)' : '';
                return '<div style="padding:6px 8px;border-bottom:1px solid var(--sl-border);font-size:0.7rem;display:flex;align-items:center;gap:8px;">'
                    + icon + '<span style="flex:1;">' + a.action + dur + '</span>'
                    + '<span style="color:var(--sl-text-disabled);">' + a.time + '</span></div>';
            }).join('');
        }

        // ─── WebSocket ───────────────────────────────────────────────
        let ws = null;
        function connectWS() {
            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(proto + '//' + location.host + '/ws');
            ws.onopen = () => {
                document.getElementById('ws-dot').className = 'pulse-dot';
                document.getElementById('ws-text').textContent = 'Connected';
            };
            ws.onclose = () => {
                document.getElementById('ws-dot').className = 'pulse-dot offline';
                document.getElementById('ws-text').textContent = 'Reconnecting...';
                setTimeout(connectWS, 3000);
            };
            ws.onmessage = (e) => {
                try {
                    const data = JSON.parse(e.data);
                    if (data.type === 'status') updateStatus(data);
                    if (data.type === 'task_update') refreshTasks();
                    if (data.type === 'workflow_update') refreshWorkflows();
                } catch(err) {}
            };
        }
        connectWS();

        function updateStatus(data) {
            const dot = document.getElementById('runner-dot');
            const text = document.getElementById('runner-status-text');
            const topDot = document.getElementById('topbar-dot');
            const topText = document.getElementById('topbar-status-text');
            if (data.runner_online) {
                dot.className = 'pulse-dot';
                text.textContent = 'Runner Online';
                if (topDot) topDot.className = 'pulse-dot';
                if (topText) topText.textContent = 'Online';
            } else {
                dot.className = 'pulse-dot offline';
                text.textContent = 'Runner Offline';
                if (topDot) topDot.className = 'pulse-dot offline';
                if (topText) topText.textContent = 'Offline';
            }
        }

        // ─── SLATE Controls (with AI Intelligence) ─────────────────
        async function slateCtrl(action, btn) {
            if (btn.classList.contains('running')) return;
            btn.classList.remove('ok', 'err');
            btn.classList.add('running');
            const term = document.getElementById('ctrl-term');
            term.classList.add('open');
            term.innerHTML = '<span class="lbl">Running ' + action + '...</span>\\n';
            const t0 = performance.now();
            try {
                const res = await fetch('/api/slate/' + action, { method: 'POST' });
                const data = await res.json();
                const elapsed = Math.round(performance.now() - t0);
                btn.classList.remove('running');
                if (data.steps) {
                    let html = '';
                    data.steps.forEach(s => {
                        const icon = s.success ? '<span class="ok">&#10003;</span>' : '<span class="err">&#10007;</span>';
                        html += icon + ' <span class="lbl">' + s.step + '</span>\\n';
                        if (s.output) html += s.output + '\\n';
                        if (s.error) html += '<span class="err">' + s.error + '</span>\\n';
                    });
                    term.innerHTML = html;
                    btn.classList.add(data.success ? 'ok' : 'err');
                } else {
                    term.innerHTML = data.success
                        ? '<span class="ok">&#10003;</span> <span class="lbl">' + action + '</span>\\n' + (data.output || 'Done')
                        : '<span class="err">&#10007;</span> <span class="lbl">' + action + '</span>\\n' + (data.error || 'Failed');
                    btn.classList.add(data.success ? 'ok' : 'err');
                }
                setDot(action.replace('run-protocol','protocol'), data.success);
                logAction(action, data.success, elapsed);
                // Record action for AI learning
                fetch('/api/slate/ai/record', { method:'POST', headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({ action: action.replace('run-',''), success: data.success, duration_ms: elapsed })
                }).catch(()=>{});
                // Request AI summary if failed
                if (!data.success) {
                    const errText = (data.steps || []).filter(s => !s.success).map(s => s.error || s.output || '').join('; ') || data.error || '';
                    fetch('/api/slate/ai/recovery', { method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({ action, error: errText })
                    }).then(r => r.json()).then(d => {
                        if (d.suggestion) term.innerHTML += '\\n<span class="lbl">&#9672; AI Fix:</span> ' + d.suggestion + '\\n';
                    }).catch(()=>{});
                }
            } catch (e) {
                btn.classList.remove('running');
                btn.classList.add('err');
                term.innerHTML = '<span class="err">&#10007; ' + e.message + '</span>';
            }
        }

        async function loadAiRecommend() {
            const el = document.getElementById('ai-rec');
            const txt = document.getElementById('ai-rec-text');
            el.style.display = 'block';
            txt.textContent = 'Analyzing system state...';
            try {
                const res = await fetch('/api/slate/ai/recommend');
                const data = await res.json();
                txt.textContent = data.recommendation || 'No recommendation available.';
            } catch (e) {
                txt.textContent = 'Run SLATE protocol to check system health.';
            }
        }

        async function slateDeploy(deployAction, btn) {
            if (btn.classList.contains('running')) return;
            btn.classList.remove('ok', 'err');
            btn.classList.add('running');
            const term = document.getElementById('ctrl-term');
            term.classList.add('open');
            term.innerHTML = '<span class="lbl">Services: ' + deployAction + '...</span>\\n';
            try {
                const res = await fetch('/api/slate/deploy/' + deployAction, { method: 'POST' });
                const data = await res.json();
                btn.classList.remove('running');
                term.innerHTML = data.success
                    ? '<span class="ok">&#10003;</span> Services ' + deployAction + '\\n' + (data.output || 'Done')
                    : '<span class="err">&#10007;</span> Services ' + deployAction + '\\n' + (data.error || 'Failed');
                btn.classList.add(data.success ? 'ok' : 'err');
                setDot('deploy', data.success);
            } catch (e) {
                btn.classList.remove('running');
                btn.classList.add('err');
                term.innerHTML = '<span class="err">&#10007; ' + e.message + '</span>';
            }
        }

        function setDot(name, ok) {
            const d = document.getElementById('dot-' + name);
            if (d) d.className = 'ctrl-dot ' + (ok ? 'ok' : 'err');
        }

        function toggleTerm() {
            document.getElementById('ctrl-term').classList.toggle('open');
        }

        // Deploy right-click for stop/status
        document.getElementById('ctrl-deploy')?.addEventListener('contextmenu', function(e) {
            e.preventDefault();
            const action = prompt('Services action:', 'stop');
            if (action && ['start','stop','status'].includes(action)) slateDeploy(action, this);
        });

        // ─── Dispatch Workflow ────────────────────────────────────────
        async function dispatchWorkflow(name) {
            try {
                const res = await fetch('/api/dispatch/' + name, { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    showToast('Dispatched: ' + name);
                    setTimeout(refreshWorkflows, 3000);
                } else {
                    showToast('Failed: ' + (data.error || 'Unknown error'), true);
                }
            } catch (e) {
                showToast('Error: ' + e.message, true);
            }
        }

        function showToast(msg, isError) {
            const term = document.getElementById('ctrl-term');
            term.classList.add('open');
            const cls = isError ? 'err' : 'ok';
            term.innerHTML += '<span class="' + cls + '">' + (isError ? '&#10007; ' : '&#10003; ') + msg + '</span>\\n';
        }

        // ─── Refresh Functions ────────────────────────────────────────
        async function refreshTasks() {
            try {
                const res = await fetch('/api/tasks');
                const data = await res.json();
                const list = document.getElementById('task-list');
                if (!data.tasks || data.tasks.length === 0) {
                    list.innerHTML = '<div class="empty">No tasks</div>';
                    return;
                }
                const navBadge = document.getElementById('nav-task-count');
                if (navBadge) navBadge.textContent = data.tasks.length;
                list.innerHTML = data.tasks.map(t => {
                    const statusCls = t.status === 'completed' ? 'online' : t.status === 'in_progress' ? 'warning' : 'pending';
                    return '<div class="task-item"><span>' + (t.title || t.id) + '</span><span class="badge ' + statusCls + '">' + t.status + '</span></div>';
                }).join('');
            } catch (e) { console.log('Tasks refresh error:', e); }
        }

        async function refreshWorkflows() {
            try {
                const res = await fetch('/api/workflows');
                const data = await res.json();
                const list = document.getElementById('workflow-list');
                if (!data.runs || data.runs.length === 0) {
                    list.innerHTML = '<div class="empty">No workflows</div>';
                    return;
                }
                const navWfBadge = document.getElementById('nav-wf-count');
                if (navWfBadge) navWfBadge.textContent = data.runs.length;
                list.innerHTML = data.runs.slice(0, 10).map(w => {
                    const icon = w.conclusion === 'success' ? '<span class="ok">&#10003;</span>' :
                                 w.conclusion === 'failure' ? '<span class="err">&#10007;</span>' :
                                 w.status === 'in_progress' ? '<span style="color:var(--sl-warning)">&#9679;</span>' : '&#9679;';
                    return '<div class="wf-item"><span class="wf-name">' + icon + ' ' + (w.name || '') + '</span><span class="wf-meta">#' + (w.run_number || '') + '</span></div>';
                }).join('');
            } catch (e) { console.log('Workflows refresh error:', e); }
        }

        async function refreshRunner() {
            try {
                const res = await fetch('/api/runner');
                const data = await res.json();
                updateStatus({ runner_online: data.status === 'online' || data.runner?.status === 'online' });
            } catch (e) {}
        }

        async function refreshWorkflowPipeline() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                if (data.pipeline) {
                    document.getElementById('flow-task-count').textContent = data.pipeline.tasks || '0';
                    document.getElementById('flow-runner-status').textContent = data.pipeline.runner || 'Idle';
                    document.getElementById('flow-workflow-count').textContent = data.pipeline.workflows || '0';
                    document.getElementById('flow-pr-count').textContent = data.pipeline.prs || '0';
                }
                if (data.tasks) {
                    document.getElementById('flow-task-count').textContent = data.tasks.total || '0';
                }
            } catch (e) {}
        }

        async function refreshGitHub() {
            try {
                const res = await fetch('/api/github');
                const data = await res.json();
                // PRs
                const prList = document.getElementById('pr-list');
                const prCount = document.getElementById('pr-count');
                if (data.prs && data.prs.length > 0) {
                    prCount.textContent = data.prs.length;
                    prList.innerHTML = data.prs.slice(0, 5).map(pr => '<div class="gh-item">#' + pr.number + ' ' + pr.title + '</div>').join('');
                } else { prList.innerHTML = '<div class="empty">No open PRs</div>'; prCount.textContent = '0'; }
                // Commits
                const commitList = document.getElementById('commit-list');
                if (data.commits && data.commits.length > 0) {
                    commitList.innerHTML = data.commits.slice(0, 5).map(c => '<div class="gh-item" style="font-family:var(--sl-font-mono);font-size:0.6rem;">' + (c.sha || '').substring(0,7) + ' ' + (c.message || '').substring(0,50) + '</div>').join('');
                } else { commitList.innerHTML = '<div class="empty">No commits</div>'; }
                // Issues
                const issueList = document.getElementById('issue-list');
                const issueCount = document.getElementById('issue-count');
                if (data.issues && data.issues.length > 0) {
                    issueCount.textContent = data.issues.length;
                    issueList.innerHTML = data.issues.slice(0, 5).map(i => '<div class="gh-item">#' + i.number + ' ' + i.title + '</div>').join('');
                } else { issueList.innerHTML = '<div class="empty">No issues</div>'; issueCount.textContent = '0'; }
                // Release
                if (data.release) {
                    document.getElementById('release-info').innerHTML = '<div class="gh-item"><strong>' + (data.release.tag || '') + '</strong> ' + (data.release.name || '') + '</div>';
                }
            } catch (e) { console.log('GitHub refresh error:', e); }
        }

        async function refreshSystemHealth() {
            try {
                const res = await fetch('/api/system/gpu');
                const data = await res.json();
                // GPU utilization
                const gpuDiv = document.getElementById('gpu-utilization');
                if (data.gpus && data.gpus.length > 0) {
                    gpuDiv.innerHTML = data.gpus.map((g, i) => {
                        return '<div class="metric"><span class="metric-label">' + (g.name || 'GPU ' + i) + '</span><div class="metric-bar"><div class="metric-fill" style="width:' + (g.utilization || 0) + '%"></div></div><span class="metric-value">' + (g.utilization || 0) + '%</span></div>' +
                               '<div style="font-size:0.6rem;color:var(--sl-text-disabled);padding:0 0 4px var(--sl-space-3);">' + (g.memory_used || 0) + ' / ' + (g.memory_total || 0) + ' MiB</div>';
                    }).join('');
                    // Update hardware panel too
                    data.gpus.forEach((g, i) => {
                        const bar = document.getElementById('hw-gpu' + i + '-bar');
                        const val = document.getElementById('hw-gpu' + i + '-val');
                        if (bar) bar.style.width = (g.utilization || 0) + '%';
                        if (val) val.textContent = (g.utilization || 0) + '%';
                    });
                }
                // System resources
                if (data.system) {
                    const cpu = data.system.cpu_percent || 0;
                    const mem = data.system.memory_percent || 0;
                    const disk = data.system.disk_percent || 0;
                    document.getElementById('cpu-bar').style.width = cpu + '%';
                    document.getElementById('cpu-value').textContent = cpu + '%';
                    document.getElementById('memory-bar').style.width = mem + '%';
                    document.getElementById('memory-value').textContent = mem + '%';
                    document.getElementById('disk-bar').style.width = disk + '%';
                    document.getElementById('disk-value').textContent = disk + '%';
                }
                // Ollama
                if (data.ollama) {
                    const ollamaDiv = document.getElementById('ollama-status');
                    if (data.ollama.models && data.ollama.models.length > 0) {
                        ollamaDiv.innerHTML = data.ollama.models.map(m => '<div style="font-size:0.7rem;color:var(--sl-text-tertiary);padding:2px 0;">' + m.name + ' <span style="color:var(--sl-text-disabled)">' + (m.size || '') + '</span></div>').join('');
                    } else {
                        ollamaDiv.innerHTML = '<div class="empty">No models loaded</div>';
                    }
                }
            } catch (e) { console.log('System health error:', e); }
        }

        async function refreshServices() {
            try {
                const res = await fetch('/api/services');
                const data = await res.json();
                if (data.services) {
                    const mapping = {
                        'orchestrator': 'orch-badge',
                        'runner': 'runner-badge',
                        'ollama': 'ollama-badge',
                        'foundry': 'foundry-badge',
                        'docker': 'docker-badge'
                    };
                    for (const [key, elemId] of Object.entries(mapping)) {
                        const badge = document.getElementById(elemId);
                        if (badge && data.services[key] !== undefined) {
                            const online = data.services[key];
                            badge.className = 'badge ' + (online ? 'online' : 'offline');
                            badge.textContent = online ? 'Online' : 'Offline';
                        }
                    }
                }
            } catch (e) {}
        }

        async function refreshActivity() {
            try {
                const res = await fetch('/api/activity');
                const data = await res.json();
                const feed = document.getElementById('activity-feed');
                if (data.events && data.events.length > 0) {
                    feed.innerHTML = data.events.slice(0, 8).map(e => '<div class="feed-item"><div class="feed-dot"></div><div><div class="feed-text">' + e.message + '</div><div class="feed-time">' + (e.time || '') + '</div></div></div>').join('');
                } else {
                    feed.innerHTML = '<div class="empty">No activity</div>';
                }
            } catch (e) {}
        }

        async function refreshHeatmap() {
            try {
                const res = await fetch('/api/heatmap');
                const data = await res.json();
                if (data.total !== undefined) {
                    document.getElementById('heatmap-total').textContent = data.total + ' tasks';
                }
                const grid = document.getElementById('heatmap-grid');
                if (data.weeks) {
                    grid.innerHTML = data.weeks.map(week => {
                        return '<div class="heatmap-week">' + week.map(day => '<div class="heatmap-day" data-level="' + (day.level || 0) + '" title="' + (day.date || '') + ': ' + (day.count || 0) + '"></div>').join('') + '</div>';
                    }).join('');
                }
            } catch (e) {}
        }

        async function refreshTechTree() {
            try {
                const res = await fetch('/api/tech-tree');
                const data = await res.json();
                const stats = document.getElementById('tech-tree-stats');
                if (data.by_status) {
                    stats.innerHTML = Object.entries(data.by_status).map(([k,v]) => '<div class="tt-stat"><div class="tt-stat-val">' + v + '</div><div class="tt-stat-lbl">' + k + '</div></div>').join('');
                } else {
                    stats.innerHTML = '<div class="empty">No tech tree data</div>';
                }
            } catch (e) {}
        }

        async function refreshAgents() {
            try {
                const res = await fetch('/api/agents');
                const data = await res.json();
                const list = document.getElementById('agent-list');
                if (data.agents && data.agents.length > 0) {
                    list.innerHTML = data.agents.map(a => {
                        const cls = a.status === 'active' ? 'online' : 'pending';
                        return '<div class="svc-item"><div class="svc-name"><div class="svc-icon">' + (a.name || '?')[0] + '</div><span>' + (a.name || 'Unknown') + '</span></div><span class="badge ' + cls + '">' + (a.role || a.status || '') + '</span></div>';
                    }).join('');
                } else {
                    list.innerHTML = '<div class="empty">No agents registered</div>';
                }
            } catch (e) {}
        }

        async function refreshForks() {
            try {
                const res = await fetch('/api/forks');
                const data = await res.json();
                const list = document.getElementById('fork-list');
                if (data.forks && data.forks.length > 0) {
                    list.innerHTML = data.forks.slice(0, 5).map(f => '<div class="gh-item">' + f.full_name + '</div>').join('');
                } else {
                    list.innerHTML = '<div class="empty">No forks</div>';
                }
            } catch (e) {}
        }

        async function refreshSpecs() {
            try {
                const res = await fetch('/api/specs');
                const data = await res.json();
                const list = document.getElementById('spec-list');
                if (data.specs && data.specs.length > 0) {
                    list.innerHTML = data.specs.slice(0, 5).map(s => {
                        const cls = s.status === 'approved' ? 'online' : s.status === 'implementing' ? 'warning' : 'pending';
                        return '<div class="svc-item"><span style="font-size:0.7rem;color:var(--sl-text-tertiary);">' + s.title + '</span><span class="badge ' + cls + '">' + s.status + '</span></div>';
                    }).join('');
                } else {
                    list.innerHTML = '<div class="empty">No specs</div>';
                }
            } catch (e) {}
        }

        async function refreshMultiRunner() {
            try {
                const res = await fetch('/api/multirunner');
                const data = await res.json();
                const el = document.getElementById('multi-runner-status');
                if (data.runners) {
                    el.innerHTML = '<div style="font-size:0.7rem;color:var(--sl-text-tertiary);">Active: ' + (data.active || 0) + ' / ' + (data.total || 0) + '</div>';
                } else {
                    el.innerHTML = '<div class="empty">Not configured</div>';
                }
            } catch (e) {
                document.getElementById('multi-runner-status').innerHTML = '<div class="empty">Not available</div>';
            }
        }

        async function refreshWorkflowStats() {
            // Placeholder for workflow analytics
        }

        async function refreshDocker() {
            try {
                const res = await fetch('/api/docker');
                const data = await res.json();
                const el = document.getElementById('docker-containers');
                if (data.containers && data.containers.length > 0) {
                    el.innerHTML = data.containers.map(c => {
                        const cls = c.state === 'running' ? 'online' : 'offline';
                        return '<div class="svc-item"><div class="svc-name"><div class="svc-icon">&#128051;</div><span style="font-size:0.7rem;">' + (c.name || '') + '</span></div><span class="badge ' + cls + '">' + (c.state || '') + '</span></div>';
                    }).join('');
                } else {
                    el.innerHTML = '<div class="empty">No containers</div>';
                }
            } catch (e) {}
        }

        async function dockerActionAll(action) {
            try {
                await fetch('/api/docker/' + action, { method: 'POST' });
                showToast('Docker ' + action + ' initiated');
                setTimeout(refreshDocker, 2000);
            } catch (e) {}
        }

        async function createTask() {
            const title = document.getElementById('new-task-title').value.trim();
            if (!title) return;
            const priority = document.getElementById('new-task-priority').value;
            try {
                await fetch('/api/tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title, priority: parseInt(priority) })
                });
                document.getElementById('new-task-title').value = '';
                refreshTasks();
            } catch (e) {}
        }

        async function runBenchmark() {
            const term = document.getElementById('ctrl-term');
            term.classList.add('open');
            term.innerHTML = '<span class="lbl">Running benchmarks...</span>\\n';
            try {
                const res = await fetch('/api/slate/benchmark', { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    term.innerHTML += '<span class="ok">&#10003;</span> ' + (data.output || 'Done') + '\\n';
                } else {
                    term.innerHTML += '<span class="err">&#10007;</span> ' + (data.error || 'Failed') + '\\n';
                }
            } catch (e) {
                term.innerHTML += '<span class="err">&#10007; ' + e.message + '</span>\\n';
            }
        }

        // ─── Refresh All ──────────────────────────────────────────────
        function refreshAll() {
            refreshTasks();
            refreshWorkflows();
            refreshRunner();
            refreshWorkflowPipeline();
            refreshGitHub();
            refreshSystemHealth();
            refreshServices();
            refreshActivity();
            refreshHeatmap();
            refreshTechTree();
            refreshAgents();
            refreshMultiRunner();
            refreshSpecs();
            refreshForks();
            refreshDocker();
        }

        // ─── Schematic Hero Widget ──────────────────────────────────────
        let schematicWs = null;
        let schematicReconnectAttempts = 0;
        const maxSchematicReconnects = 5;

        async function loadSchematic() {
            const container = document.getElementById('schematic-svg-container');
            if (!container) return;

            try {
                const res = await fetch('/api/schematic/system-state?format=svg');
                if (res.ok) {
                    const data = await res.json();
                    container.innerHTML = data.svg || '';
                } else {
                    container.innerHTML = '<div class="schematic-loading"><span>Schematic unavailable</span></div>';
                }
            } catch (e) {
                container.innerHTML = '<div class="schematic-loading"><span>Schematic API offline</span></div>';
            }
        }

        function connectSchematicWebSocket() {
            if (schematicWs && schematicWs.readyState === WebSocket.OPEN) return;

            try {
                schematicWs = new WebSocket(`ws://${window.location.host}/api/schematic/ws/live`);

                schematicWs.onopen = () => {
                    schematicReconnectAttempts = 0;
                    console.log('[Schematic] WebSocket connected');
                };

                schematicWs.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === 'schematic_update') {
                            const container = document.getElementById('schematic-svg-container');
                            if (container && data.svg) {
                                container.innerHTML = data.svg;
                            }
                        }
                    } catch (e) {}
                };

                schematicWs.onclose = () => {
                    schematicWs = null;
                    if (schematicReconnectAttempts < maxSchematicReconnects) {
                        schematicReconnectAttempts++;
                        setTimeout(connectSchematicWebSocket, 5000);
                    }
                };

                schematicWs.onerror = () => {
                    schematicWs?.close();
                };
            } catch (e) {}
        }

        function requestSchematicUpdate() {
            if (schematicWs && schematicWs.readyState === WebSocket.OPEN) {
                schematicWs.send(JSON.stringify({ type: 'request_update', timestamp: Date.now() }));
            }
        }

        // ─── Schematic Widget Library (Spec 012 Phase 2) ──────────────

        // Modal detail view
        function openSchematicModal(templateId) {
            const modal = document.getElementById('schematic-modal');
            const body = document.getElementById('schematic-modal-body');
            const title = document.getElementById('schematic-modal-title');

            if (!modal || !body) return;

            const titles = {
                'system': 'SLATE System Architecture',
                'inference': 'GPU Inference Pipeline',
                'ci-cd': 'CI/CD Pipeline',
                'agents': 'Agent Routing System'
            };

            title.textContent = titles[templateId] || 'Schematic Detail';
            body.innerHTML = '<div class="schematic-loading"><div class="schematic-loading-spinner"></div><span>Loading...</span></div>';
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';

            // Fetch the template schematic
            fetch(`/api/schematic/template/${templateId}`)
                .then(r => r.ok ? r.json() : Promise.reject('Template not found'))
                .then(data => {
                    body.innerHTML = data.svg || '<div class="schematic-loading"><span>No schematic data</span></div>';
                    document.getElementById('schematic-modal-timestamp').textContent =
                        'Last updated: ' + new Date().toLocaleTimeString();
                })
                .catch(() => {
                    // Fall back to system widget
                    fetch('/api/schematic/widget/system')
                        .then(r => r.ok ? r.json() : Promise.reject())
                        .then(data => {
                            body.innerHTML = data.html || '';
                            document.getElementById('schematic-modal-timestamp').textContent =
                                'Last updated: ' + new Date().toLocaleTimeString();
                        })
                        .catch(() => {
                            body.innerHTML = '<div class="schematic-loading"><span>Schematic unavailable</span></div>';
                        });
                });
        }

        function closeSchematicModal() {
            const modal = document.getElementById('schematic-modal');
            if (modal) {
                modal.classList.remove('active');
                document.body.style.overflow = '';
            }
        }

        // Close modal on Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeSchematicModal();
        });

        // Close modal on backdrop click
        document.getElementById('schematic-modal')?.addEventListener('click', function(e) {
            if (e.target === this) closeSchematicModal();
        });

        // Sidebar compact widget updater
        async function updateSidebarSchematic() {
            try {
                const res = await fetch('/api/schematic/widget/compact?template_id=system');
                if (res.ok) {
                    const data = await res.json();
                    const container = document.getElementById('sidebar-schematic-svg');
                    if (container && data.html) {
                        container.innerHTML = data.html;
                    }
                }
            } catch (e) {}
        }

        // Status overlay updater
        async function updateSchematicStatusOverlay() {
            try {
                const res = await fetch('/api/schematic/system-state');
                if (res.ok) {
                    const data = await res.json();
                    // Update agent status chips based on live data
                    const chips = document.querySelectorAll('.schematic-status-chip');
                    if (data.components && chips.length > 0) {
                        const componentNames = Object.keys(data.components);
                        chips.forEach((chip, i) => {
                            const dot = chip.querySelector('.ssc-dot');
                            if (dot && componentNames[i]) {
                                const status = data.components[componentNames[i]]?.status || 'idle';
                                dot.className = 'ssc-dot ' + (status === 'active' ? 'active' : status === 'warning' ? 'warning' : status === 'error' ? 'error' : 'idle');
                            }
                        });
                    }
                }
            } catch (e) {}
        }

        // ─── Schematic Interactive Features (Spec 012 Phase 3) ─────────

        // Component data for tooltips
        const schematicComponents = {
            'dashboard': { name: 'Dashboard', desc: 'Web UI at :8080', status: 'active', port: '8080' },
            'ollama': { name: 'Ollama', desc: 'Local LLM inference', status: 'active', port: '11434' },
            'foundry': { name: 'Foundry Local', desc: 'ONNX inference backend', status: 'idle', port: '5272' },
            'chromadb': { name: 'ChromaDB', desc: 'Vector store for RAG', status: 'idle', port: '8000' },
            'runner': { name: 'slate-runner', desc: 'GitHub Actions runner', status: 'active' },
            'gpu-cluster': { name: 'Dual GPU', desc: 'RTX 5070 Ti x2', status: 'active' },
            'task-router': { name: 'Task Router', desc: 'Agentic dispatch system', status: 'active' }
        };

        // Tooltip system
        const tooltip = document.getElementById('schematic-tooltip');
        const tooltipTitle = document.getElementById('tooltip-title');
        const tooltipDesc = document.getElementById('tooltip-desc');
        const tooltipDot = document.getElementById('tooltip-status-dot');
        const tooltipStatus = document.getElementById('tooltip-status-text');

        function showSchematicTooltip(componentId, x, y) {
            const comp = schematicComponents[componentId];
            if (!comp || !tooltip) return;

            tooltipTitle.textContent = comp.name;
            tooltipDesc.textContent = comp.desc + (comp.port ? ` (:${comp.port})` : '');

            const statusColors = { active: '#22C55E', warning: '#F59E0B', error: '#EF4444', idle: '#6B7280' };
            tooltipDot.style.background = statusColors[comp.status] || statusColors.idle;
            tooltipStatus.textContent = comp.status.charAt(0).toUpperCase() + comp.status.slice(1);

            // Position tooltip
            const rect = tooltip.getBoundingClientRect();
            let left = x + 12;
            let top = y - 8;

            if (left + 240 > window.innerWidth) left = x - 250;
            if (top + 80 > window.innerHeight) top = y - 80;

            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
            tooltip.classList.add('visible');
        }

        function hideSchematicTooltip() {
            if (tooltip) tooltip.classList.remove('visible');
        }

        // Attach tooltip listeners to schematic SVG nodes
        function attachSchematicInteractivity() {
            document.querySelectorAll('.schematic-content svg, .schematic-modal-body svg').forEach(svg => {
                // Find nodes (g elements or rects with data-component)
                svg.querySelectorAll('[data-component]').forEach(node => {
                    node.style.cursor = 'pointer';
                    node.classList.add('schematic-node');

                    node.addEventListener('mouseenter', function(e) {
                        const compId = this.getAttribute('data-component');
                        showSchematicTooltip(compId, e.clientX, e.clientY);
                        highlightConnections(compId, svg);
                    });

                    node.addEventListener('mousemove', function(e) {
                        const compId = this.getAttribute('data-component');
                        showSchematicTooltip(compId, e.clientX, e.clientY);
                    });

                    node.addEventListener('mouseleave', function() {
                        hideSchematicTooltip();
                        clearConnectionHighlights(svg);
                    });
                });
            });
        }

        // Connection highlighting
        function highlightConnections(componentId, svg) {
            svg.querySelectorAll('[data-from], [data-to]').forEach(conn => {
                const from = conn.getAttribute('data-from');
                const to = conn.getAttribute('data-to');
                if (from === componentId || to === componentId) {
                    conn.classList.add('highlighted', 'schematic-flow-active');
                }
            });

            svg.querySelectorAll('[data-component]').forEach(node => {
                if (node.getAttribute('data-component') === componentId) {
                    node.classList.add('highlighted');
                }
            });
        }

        function clearConnectionHighlights(svg) {
            svg.querySelectorAll('.highlighted').forEach(el => {
                el.classList.remove('highlighted', 'schematic-flow-active');
            });
        }

        // Zoom functionality
        let currentZoom = 1;
        const zoomStep = 0.25;
        const maxZoom = 2.5;
        const minZoom = 0.5;

        function zoomSchematic(delta) {
            const container = document.querySelector('.schematic-modal-body svg, .schematic-hero svg');
            if (!container) return;

            currentZoom = Math.max(minZoom, Math.min(maxZoom, currentZoom + delta));
            container.style.transform = `scale(${currentZoom})`;

            if (currentZoom > 1) {
                container.classList.add('zoomed');
            } else {
                container.classList.remove('zoomed');
            }
        }

        function resetZoom() {
            currentZoom = 1;
            const container = document.querySelector('.schematic-modal-body svg, .schematic-hero svg');
            if (container) {
                container.style.transform = 'scale(1)';
                container.classList.remove('zoomed');
            }
        }

        // Re-attach interactivity when schematics update
        const schematicObserver = new MutationObserver(() => {
            attachSchematicInteractivity();
        });

        ['schematic-svg-container', 'schematic-modal-body', 'sidebar-schematic-svg'].forEach(id => {
            const el = document.getElementById(id);
            if (el) schematicObserver.observe(el, { childList: true, subtree: true });
        });

        // Initial attachment
        setTimeout(attachSchematicInteractivity, 1000);

        // ─── Initial Load ─────────────────────────────────────────────
        async function fetchInitialStatus() {
            try {
                const res = await fetch('/api/orchestrator');
                const data = await res.json();
                updateStatus(data);
            } catch (e) {}
        }

        // ─── Watchmaker Phase 3: View Update Functions ───────────
        // Modified: 2026-02-08T06:00:00Z | Author: COPILOT | Change: Add watchmaker dashboard view controllers

        async function updateHealthRing() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                // GPU
                const gpuCount = (data.gpus || []).length;
                const gpuOk = gpuCount > 0;
                setEl('hr-gpu-val', gpuCount + 'x GPU');
                setJewel('hr-gpu-jewel', gpuOk ? 'active' : 'error');
                // AI (Ollama models)
                const modelCount = (data.ollama_models || []).length;
                setEl('hr-ai-val', modelCount + ' models');
                setJewel('hr-ai-jewel', modelCount > 0 ? 'active' : 'inactive');
                // Services
                const svcTotal = (data.services_online || 0);
                setEl('hr-svc-val', svcTotal + ' online');
                setJewel('hr-svc-jewel', svcTotal >= 3 ? 'active' : svcTotal > 0 ? 'pending' : 'error');
                // Command center summary cards
                setEl('cc-svc-count', svcTotal + '/' + (data.services_total || '?'));
                setJewel('cc-svc-jewel', svcTotal >= 3 ? 'active' : 'pending');
                setEl('cc-model-count', modelCount.toString());
                setJewel('cc-model-jewel', modelCount > 0 ? 'active' : 'inactive');
            } catch (e) {}
            try {
                const resp2 = await fetch('/api/tasks');
                const tasks = await resp2.json();
                const pending = (tasks.tasks || []).filter(t => t.status === 'pending' || t.status === 'not_started').length;
                const running = (tasks.tasks || []).filter(t => t.status === 'in_progress' || t.status === 'running').length;
                setEl('hr-tasks-val', pending + ' pending');
                setJewel('hr-tasks-jewel', running > 0 ? 'active' : pending > 0 ? 'pending' : 'inactive');
                setEl('cc-task-count', pending.toString());
                setJewel('cc-task-jewel', pending > 0 ? 'pending' : 'active');
            } catch (e) {
                setEl('hr-tasks-val', '--');
            }
            try {
                const resp3 = await fetch('/api/runner/status');
                const r = await resp3.json();
                const rOnline = r.process_running || r.runner_running || false;
                setEl('hr-runner-val', rOnline ? 'Online' : 'Offline');
                setJewel('hr-runner-jewel', rOnline ? 'active' : 'inactive');
            } catch (e) {
                setEl('hr-runner-val', '--');
            }
        }

        function setEl(id, text) {
            const el = document.getElementById(id);
            if (el) el.textContent = text;
        }

        function setJewel(id, status) {
            const el = document.getElementById(id);
            if (!el) return;
            el.className = 'status-jewel ' + status;
        }

        async function updateConstellationJewels() {
            try {
                const resp = await fetch('/api/services');
                const data = await resp.json();
                const svcMap = {};
                (data.services || []).forEach(s => { svcMap[s.name.toLowerCase()] = s.status; });
                const map = {
                    'cst-dash-jewel': 'dashboard',
                    'cst-ollama-jewel': 'ollama',
                    'cst-chroma-jewel': 'chromadb',
                    'cst-foundry-jewel': 'foundry',
                    'cst-runner-jewel': 'runner',
                    'cst-gpu-jewel': 'gpu'
                };
                for (const [id, svc] of Object.entries(map)) {
                    const el = document.getElementById(id);
                    if (!el) continue;
                    const status = svcMap[svc];
                    el.setAttribute('fill', status === 'online' ? '#22C55E' : status === 'offline' ? '#EF4444' : '#F59E0B');
                }
                const count = Object.values(svcMap).filter(v => v === 'online').length;
                setEl('constellation-count', count + ' / ' + Object.keys(svcMap).length + ' online');
            } catch (e) {}
        }

        async function updateGPUWorkbench() {
            try {
                const resp = await fetch('/api/gpu');
                const data = await resp.json();
                const gpus = data.gpus || [];
                gpus.forEach((gpu, i) => {
                    const util = gpu.utilization || gpu.load || 0;
                    const memPct = gpu.memory_used && gpu.memory_total ? Math.round((gpu.memory_used / gpu.memory_total) * 100) : 0;
                    const power = gpu.power_draw || gpu.power || 0;
                    const temp = gpu.temperature || gpu.temp || 0;
                    // Meters
                    const compFill = document.getElementById('wb-gpu' + i + '-compute');
                    if (compFill) compFill.style.width = util + '%';
                    setEl('wb-gpu' + i + '-compute-val', util + '%');
                    const memFill = document.getElementById('wb-gpu' + i + '-memory');
                    if (memFill) memFill.style.width = memPct + '%';
                    setEl('wb-gpu' + i + '-memory-val', memPct + '%');
                    const pwrFill = document.getElementById('wb-gpu' + i + '-power');
                    if (pwrFill) pwrFill.style.width = Math.min(100, (power / 250) * 100) + '%';
                    setEl('wb-gpu' + i + '-power-val', power + 'W');
                    setEl('wb-gpu' + i + '-temp', 'Temp: ' + temp + '°C');
                    setJewel('wb-gpu' + i + '-jewel', util > 80 ? 'pending' : 'active');
                    // Load distribution
                    const loadFill = document.getElementById('wb-load-' + i);
                    if (loadFill) loadFill.style.width = util + '%';
                    // Tasks
                    const taskEl = document.getElementById('wb-gpu' + i + '-tasks');
                    if (taskEl && gpu.processes) {
                        taskEl.innerHTML = gpu.processes.map(p =>
                            '<div class="gpu-task-item"><span class="status-jewel active" style="width:6px;height:6px;"></span> ' + (p.name || p) + '</div>'
                        ).join('') || '<div class="gpu-task-item" style="color:var(--sl-text-disabled);">&#8212; idle</div>';
                    }
                });
            } catch (e) {}
        }

        async function refreshTaskOrchestration() {
            try {
                const resp = await fetch('/api/tasks');
                const data = await resp.json();
                const tasks = data.tasks || [];
                let pending = 0, running = 0, review = 0, completed = 0;
                tasks.forEach(t => {
                    const s = (t.status || '').toLowerCase();
                    if (s === 'pending' || s === 'not_started') pending++;
                    else if (s === 'in_progress' || s === 'running') running++;
                    else if (s === 'review' || s === 'waiting') review++;
                    else if (s === 'completed' || s === 'done') completed++;
                });
                setEl('tp-pending-count', pending.toString());
                setEl('tp-running-count', running.toString());
                setEl('tp-review-count', review.toString());
                setEl('tp-completed-count', completed.toString());
                // Highlight current stage
                ['tp-pending','tp-running','tp-review','tp-completed'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) el.classList.remove('current');
                });
                if (running > 0) document.getElementById('tp-running')?.classList.add('current');
                else if (pending > 0) document.getElementById('tp-pending')?.classList.add('current');
                else document.getElementById('tp-completed')?.classList.add('current');
            } catch (e) {}
        }

        async function refreshWorkflowRuns() {
            try {
                const resp = await fetch('/api/workflows/runs');
                const data = await resp.json();
                const runs = (data.runs || data.workflow_runs || []).slice(0, 10);
                const el = document.getElementById('task-orch-runs');
                if (!el) return;
                if (runs.length === 0) { el.innerHTML = '<div class="empty">No recent workflow runs</div>'; return; }
                el.innerHTML = runs.map(r => {
                    const status = r.status || r.conclusion || 'unknown';
                    const jewel = status === 'success' || status === 'completed' ? 'active' : status === 'in_progress' || status === 'queued' ? 'pending' : status === 'failure' ? 'error' : 'inactive';
                    return '<div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.03);font-size:0.7rem;">' +
                        '<span class="status-jewel ' + jewel + '" style="width:8px;height:8px;"></span>' +
                        '<span style="flex:1;color:var(--sl-text-primary);">' + (r.name || r.workflow || 'Workflow') + '</span>' +
                        '<span style="font-family:var(--sl-font-mono);color:var(--sl-text-disabled);">#' + (r.run_number || r.id || '--') + '</span>' +
                        '<span class="badge ' + (jewel === 'active' ? 'online' : jewel === 'error' ? 'error' : 'pending') + '">' + status + '</span>' +
                        '</div>';
                }).join('');
            } catch (e) {}
        }

        // ─── Watchmaker Phase 4: Animation System ──────────────────────
        // Modified: 2026-02-08T08:00:00Z | Author: COPILOT | Change: Gear rotation controller, flow/jewel animation, card hover depth, gear parallax

        // --- Gear Rotation Controller (speed-aware) ---
        const GearController = {
            _systemLoad: 0,
            _lastSpeedClass: 'gear-speed-idle',
            _gears: [],

            init() {
                this._gears = document.querySelectorAll('.gear-bg svg.gear-animated, .watchmaker-card .gear-icon');
                this.updateSpeed(0);
            },

            // Map load percentage to speed tier
            loadToSpeedClass(load) {
                if (load <= 0) return 'gear-speed-idle';
                if (load < 20) return 'gear-speed-low';
                if (load < 50) return 'gear-speed-medium';
                if (load < 80) return 'gear-speed-high';
                return 'gear-speed-max';
            },

            updateSpeed(load) {
                this._systemLoad = load;
                const newClass = this.loadToSpeedClass(load);
                if (newClass === this._lastSpeedClass) return;

                const speedClasses = ['gear-speed-idle','gear-speed-low','gear-speed-medium','gear-speed-high','gear-speed-max'];
                this._gears.forEach(g => {
                    speedClasses.forEach(c => g.classList.remove(c));
                    g.classList.add(newClass);
                });
                this._lastSpeedClass = newClass;
            },

            pause() {
                this._gears.forEach(g => g.classList.add('gear-paused'));
            },

            resume() {
                this._gears.forEach(g => g.classList.remove('gear-paused'));
            },

            // Fetch system load and update gears
            async syncWithSystem() {
                try {
                    const resp = await fetch('/api/gpu');
                    const data = await resp.json();
                    const gpus = data.gpus || [];
                    const avgLoad = gpus.length > 0
                        ? gpus.reduce((sum, g) => sum + (g.utilization || g.load || 0), 0) / gpus.length
                        : 0;
                    this.updateSpeed(avgLoad);
                } catch (e) {
                    this.updateSpeed(0);
                }
            }
        };

        // --- Flow Line Pulse Controller ---
        const FlowController = {
            _lines: [],

            init() {
                this._lines = document.querySelectorAll('.data-flow-line');
            },

            // Set flow speed class based on throughput
            setFlowSpeed(lineEl, speed) {
                lineEl.classList.remove('flow-fast', 'flow-medium', 'flow-slow', 'flow-inactive', 'flow-glow');
                if (speed === 'fast') lineEl.classList.add('flow-fast', 'flow-glow');
                else if (speed === 'medium') lineEl.classList.add('flow-medium');
                else if (speed === 'slow') lineEl.classList.add('flow-slow');
                else lineEl.classList.add('flow-inactive');
            },

            // Update constellation flow lines based on service status
            async syncWithServices() {
                try {
                    const resp = await fetch('/api/services');
                    const data = await resp.json();
                    const svcMap = {};
                    (data.services || []).forEach(s => { svcMap[s.name.toLowerCase()] = s.status; });
                    const allOnline = Object.values(svcMap).filter(v => v === 'online').length;
                    const total = Object.keys(svcMap).length || 1;
                    const ratio = allOnline / total;
                    this._lines.forEach(l => {
                        if (ratio >= 0.8) this.setFlowSpeed(l, 'fast');
                        else if (ratio >= 0.5) this.setFlowSpeed(l, 'medium');
                        else if (ratio > 0) this.setFlowSpeed(l, 'slow');
                        else this.setFlowSpeed(l, 'inactive');
                    });
                } catch (e) {}
            }
        };

        // --- Jewel Pulse Controller ---
        const JewelController = {
            // Set pulse variant based on urgency
            setPulseVariant(jewelEl, variant) {
                jewelEl.classList.remove('pulse-urgent', 'pulse-slow', 'breathe');
                if (variant === 'urgent') jewelEl.classList.add('pulse-urgent');
                else if (variant === 'slow') jewelEl.classList.add('pulse-slow');
                else if (variant === 'breathe') jewelEl.classList.add('breathe');
            }
        };

        // --- Card Hover Depth Controller ---
        const CardDepthController = {
            _cards: [],
            _enabled: true,

            init() {
                this._cards = document.querySelectorAll('.watchmaker-card');
                // Check for reduced motion preference
                if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                    this._enabled = false;
                    return;
                }
                this._cards.forEach(card => {
                    card.addEventListener('mouseenter', (e) => this._onEnter(card, e));
                    card.addEventListener('mousemove', (e) => this._onMove(card, e));
                    card.addEventListener('mouseleave', (e) => this._onLeave(card, e));
                });
            },

            _onEnter(card) {
                card.classList.add('depth-tracking');
                card.style.position = 'relative';
                // Trigger shimmer on hover
                card.classList.add('shimmer-active');
                setTimeout(() => card.classList.remove('shimmer-active'), 2000);
            },

            _onMove(card, e) {
                if (!this._enabled) return;
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const centerX = rect.width / 2;
                const centerY = rect.height / 2;
                // Calculate rotation angles (max 3 degrees)
                const rotateY = ((x - centerX) / centerX) * 3;
                const rotateX = ((centerY - y) / centerY) * 3;
                // Shadow offset follows mouse
                const shadowX = ((x - centerX) / centerX) * 6;
                const shadowY = ((y - centerY) / centerY) * 6;
                card.style.setProperty('--card-rx', rotateX + 'deg');
                card.style.setProperty('--card-ry', rotateY + 'deg');
                card.style.setProperty('--card-tz', '16px');
                card.style.setProperty('--card-shadow-x', shadowX + 'px');
                card.style.setProperty('--card-shadow-y', shadowY + 'px');
            },

            _onLeave(card) {
                card.classList.remove('depth-tracking');
                card.style.setProperty('--card-rx', '0deg');
                card.style.setProperty('--card-ry', '0deg');
                card.style.setProperty('--card-tz', '0px');
                card.style.setProperty('--card-shadow-x', '0px');
                card.style.setProperty('--card-shadow-y', '4px');
            }
        };

        // --- Background Gear Parallax Controller ---
        const GearParallax = {
            _layers: [],
            _enabled: true,
            _ticking: false,

            init() {
                if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
                    this._enabled = false;
                    return;
                }
                this._layers = [
                    { el: document.querySelector('.gear-layer-1'), speed: 0.02 },
                    { el: document.querySelector('.gear-layer-2'), speed: 0.04 },
                    { el: document.querySelector('.gear-layer-3'), speed: 0.01 },
                    { el: document.querySelector('.gear-layer-4'), speed: 0.05 },
                    { el: document.querySelector('.gear-layer-5'), speed: 0.03 }
                ].filter(l => l.el);

                // Only use mouse-based parallax (no scroll since dashboard uses sections)
                document.addEventListener('mousemove', (e) => this._onMouseMove(e));
            },

            _onMouseMove(e) {
                if (!this._enabled || this._ticking) return;
                this._ticking = true;
                requestAnimationFrame(() => {
                    const cx = window.innerWidth / 2;
                    const cy = window.innerHeight / 2;
                    const dx = e.clientX - cx;
                    const dy = e.clientY - cy;
                    this._layers.forEach(layer => {
                        const x = dx * layer.speed;
                        const y = dy * layer.speed;
                        layer.el.style.transform = `translate(${x}px, ${y}px)`;
                    });
                    this._ticking = false;
                });
            }
        };

        // --- Animation System Initialization ---
        function initAnimationSystem() {
            GearController.init();
            FlowController.init();
            CardDepthController.init();
            GearParallax.init();
            // Sync gear speed with system load every 10s
            GearController.syncWithSystem();
            setInterval(() => GearController.syncWithSystem(), 10000);
            // Sync flow lines with service status every 30s
            FlowController.syncWithServices();
            setInterval(() => FlowController.syncWithServices(), 30000);
        }

        // Initialize watchmaker views
        function initWatchmakerViews() {
            updateHealthRing();
            updateConstellationJewels();
            updateGPUWorkbench();
            refreshTaskOrchestration();
            refreshWorkflowRuns();
        }

        fetchInitialStatus();
        refreshAll();
        loadSchematic();
        connectSchematicWebSocket();
        updateSidebarSchematic();
        updateSchematicStatusOverlay();
        setInterval(refreshAll, 15000);
        setInterval(requestSchematicUpdate, 30000);
        setInterval(updateSidebarSchematic, 60000);
        setInterval(updateSchematicStatusOverlay, 15000);
        // Watchmaker Phase 3 - periodic view updates
        initWatchmakerViews();
        setInterval(updateHealthRing, 20000);
        setInterval(updateGPUWorkbench, 10000);
        setInterval(updateConstellationJewels, 30000);
        setInterval(refreshTaskOrchestration, 30000);
        // Watchmaker Phase 4 - animation system
        initAnimationSystem();
    </script>
</body>
</html>'''


def get_full_template() -> str:
    """Return the complete HTML template string with control panel."""
    from slate_web.control_panel_ui import get_complete_control_panel

    base_template = build_template()
    js_template = build_template_js()

    # Inject practical control panel before closing </body>
    # Button-driven interface with real commands and step-by-step guidance
    control_panel = get_complete_control_panel()

    # Insert before </body></html>
    combined = base_template + js_template
    insert_point = combined.rfind('</body>')
    if insert_point > 0:
        combined = combined[:insert_point] + control_panel + combined[insert_point:]

    return combined


if __name__ == "__main__":
    html = get_full_template()
    out = Path(__file__).parent / "generated" / "dashboard-preview.html"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[SLATE Template] Generated -> {out} ({len(html)} bytes)")
