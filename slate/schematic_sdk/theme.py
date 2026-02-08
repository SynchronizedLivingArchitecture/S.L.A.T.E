"""
SLATE Schematic SDK - Theme Integration
# Modified: 2026-02-08T02:00:00Z | Author: COPILOT | Change: Unified tokens with Spec 014 watchmaker golden ratio

Integrates with locked design tokens (v4.0.0) for consistent theming.
Part of SLATE Generative UI protocols — Watchmaker + Golden Ratio (Spec 014).
ISO 128 · IEC 60617 · ASME Y14.44 · φ = 1.618
"""

from dataclasses import dataclass, field
from typing import Dict, Literal
import sys
from pathlib import Path

# Add workspace root for imports
WORKSPACE_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

# Golden ratio constant — governs all proportions
PHI = 1.6180339887


@dataclass
class SchematicColors:
    """Color palette for schematic rendering — unified with Spec 014."""
    # Background (watchmaker case texture)
    background: str = "#0D1B2A"
    background_gradient_end: str = "#1B3A4B"

    # Grid (engineering drawing — ISO 128)
    grid_lines: str = "#1B3A4B"
    grid_opacity: float = 0.3

    # Primary accent — copper (the watchmaker's metal — Spec 014)
    primary: str = "#B87333"
    primary_light: str = "#C9956B"
    primary_dark: str = "#8B5E2B"

    # Blueprint accents
    blueprint_accent: str = "#98C1D9"
    blueprint_node: str = "#E0FBFC"

    # Surface colors (watchmaker case)
    surface: str = "#0a0a0a"
    surface_variant: str = "#111111"
    surface_container: str = "#141210"
    surface_container_high: str = "#1a1816"
    surface_elevated: str = "#222020"

    # Status colors — unified Spec 014
    status_active: str = "#22C55E"
    status_pending: str = "#D4A054"
    status_error: str = "#C47070"
    status_inactive: str = "#78716C"
    status_info: str = "#7EA8BE"

    # Component type colors (IEC 60617 conventions)
    service_fill: str = "#1a1510"
    database_fill: str = "#101520"
    gpu_fill: str = "#15120a"
    ai_fill: str = "#0a1515"
    api_fill: str = "#1a2436"
    queue_fill: str = "#2a2436"
    external_fill: str = "#151015"

    # Text colors — warm white / natural earth
    text_primary: str = "#F5F0EB"
    text_secondary: str = "#A8A29E"
    text_muted: str = "#78716C"
    text_disabled: str = "#44403C"

    # Border colors
    border_default: str = "rgba(255,255,255,0.08)"
    border_variant: str = "rgba(255,255,255,0.12)"
    border_hover: str = "#B87333"

    # Connection colors (trace signals — ISO 128)
    connection_default: str = "#B87333"
    connection_active: str = "#22C55E"
    connection_muted: str = "#78716C"

    # Engineering trace colors (IEC 60617)
    trace_signal: str = "#B87333"
    trace_data: str = "#7EA8BE"
    trace_power: str = "#C47070"
    trace_control: str = "#D4A054"
    trace_ground: str = "#78716C"

    # Watchmaker jewel colors
    jewel_green: str = "#22C55E"
    jewel_amber: str = "#D4A054"
    jewel_red: str = "#C47070"
    jewel_blue: str = "#7EA8BE"

    # Dev cycle stage colors
    stage_plan: str = "#7EA8BE"
    stage_code: str = "#B87333"
    stage_test: str = "#D4A054"
    stage_deploy: str = "#78B89A"
    stage_feedback: str = "#9B89B3"


@dataclass
class SchematicTypography:
    """Typography settings — φ-derived scale (Golden Ratio)."""
    font_display: str = "'Inter Tight', 'Segoe UI', system-ui, sans-serif"
    font_body: str = "'Inter', 'Segoe UI', system-ui, sans-serif"
    font_mono: str = "'Cascadia Code', 'JetBrains Mono', 'Consolas', monospace"
    font_schematic: str = "'Consolas', 'Courier New', monospace"

    # Font sizes — φ-derived: 8, 11, 13, 16, 21, 34
    display_size: int = 34
    title_size: int = 21
    subtitle_size: int = 16
    label_size: int = 13
    sublabel_size: int = 11
    badge_size: int = 11
    caption_size: int = 8

    # Font weights
    weight_bold: int = 700
    weight_semibold: int = 600
    weight_regular: int = 400


@dataclass
class SchematicEffects:
    """Visual effects — Fibonacci spacing (1,2,3,5,8,13,21,34)."""
    # Shadow
    shadow_dx: int = 2
    shadow_dy: int = 4
    shadow_blur: int = 4
    shadow_color: str = "#000000"
    shadow_opacity: float = 0.3

    # Glow
    glow_blur: int = 2

    # Border radius — Fibonacci
    radius_xs: int = 3
    radius_sm: int = 5
    radius_md: int = 8
    radius_lg: int = 13
    radius_xl: int = 21


@dataclass
class SchematicTheme:
    """Complete theme configuration for schematics."""
    name: str = "blueprint"
    colors: SchematicColors = field(default_factory=SchematicColors)
    typography: SchematicTypography = field(default_factory=SchematicTypography)
    effects: SchematicEffects = field(default_factory=SchematicEffects)


class ThemeManager:
    """Manages schematic themes and integrates with design tokens."""

    # Pre-defined themes — all unified with Spec 014 copper palette
    THEMES: Dict[str, Dict] = {
        "blueprint": {
            "background": "#0D1B2A",
            "background_gradient_end": "#1B3A4B",
            "grid_lines": "#1B3A4B",
            "primary": "#B87333",
            "primary_light": "#C9956B",
            "primary_dark": "#8B5E2B",
        },
        "dark": {
            "background": "#0a0a0a",
            "background_gradient_end": "#141210",
            "grid_lines": "rgba(255,255,255,0.08)",
            "primary": "#B87333",
            "primary_light": "#C9956B",
            "primary_dark": "#8B5E2B",
            "surface": "#0a0a0a",
            "surface_container": "#141210",
            "surface_elevated": "#222020",
        },
        "light": {
            "background": "#FBF8F6",
            "background_gradient_end": "#F0EBE7",
            "grid_lines": "#E4E0DC",
            "primary": "#B87333",
            "primary_light": "#C9956B",
            "primary_dark": "#8B5E2B",
            "text_primary": "#1C1B1A",
            "text_secondary": "#4D4845",
            "surface_container": "#F5F2F0",
        },
    }

    def __init__(self, theme_name: Literal["blueprint", "dark", "light"] = "blueprint"):
        self.theme_name = theme_name
        self.theme = self._build_theme(theme_name)

    def _build_theme(self, theme_name: str) -> SchematicTheme:
        """Build theme from name, applying overrides."""
        colors = SchematicColors()
        typography = SchematicTypography()
        effects = SchematicEffects()

        # Apply theme-specific overrides
        if theme_name in self.THEMES:
            overrides = self.THEMES[theme_name]
            for key, value in overrides.items():
                if hasattr(colors, key):
                    setattr(colors, key, value)

        return SchematicTheme(
            name=theme_name,
            colors=colors,
            typography=typography,
            effects=effects
        )

    def get_status_color(self, status: str) -> str:
        """Get color for component status."""
        status_map = {
            "active": self.theme.colors.status_active,
            "pending": self.theme.colors.status_pending,
            "error": self.theme.colors.status_error,
            "inactive": self.theme.colors.status_inactive,
        }
        return status_map.get(status, self.theme.colors.status_inactive)

    def get_component_fill(self, component_type: str) -> str:
        """Get fill color for component type."""
        fill_map = {
            "service": self.theme.colors.service_fill,
            "database": self.theme.colors.database_fill,
            "gpu": self.theme.colors.gpu_fill,
            "ai": self.theme.colors.ai_fill,
            "api": self.theme.colors.api_fill,
            "queue": self.theme.colors.queue_fill,
            "external": self.theme.colors.external_fill,
        }
        return fill_map.get(component_type, self.theme.colors.surface_container)

    def get_component_border(self, component_type: str, status: str = "active") -> str:
        """Get border color for component."""
        if status == "active":
            return self.theme.colors.primary
        return self.get_status_color(status)


# Pre-instantiated themes for convenience
BlueprintTheme = lambda: ThemeManager("blueprint").theme
DarkTheme = lambda: ThemeManager("dark").theme
LightTheme = lambda: ThemeManager("light").theme
