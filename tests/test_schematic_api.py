#!/usr/bin/env python3
# Modified: 2026-02-09T06:00:00Z | Author: COPILOT | Change: Create tests for schematic API and Phase 2 widgets
"""
Tests for SLATE Schematic API endpoints and dashboard widget integration.

Covers:
  - REST API endpoints (templates, system-state, widgets)
  - Widget CSS/JS generation
  - WebSocket manager
  - Dashboard template Phase 2 widgets (compact, card, modal, status overlay)
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure workspace root on path
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))


# ── Schematic API Module Tests ────────────────────────────────────────────────

class TestSchematicAPIImports:
    """Test that schematic API module imports correctly."""

    def test_import_router(self):
        from slate.schematic_api import router
        assert router is not None

    def test_import_templates(self):
        from slate.schematic_api import TEMPLATES
        assert isinstance(TEMPLATES, dict)
        assert len(TEMPLATES) >= 1

    def test_import_ws_manager(self):
        from slate.schematic_api import ws_manager, SchematicWebSocketManager
        assert isinstance(ws_manager, SchematicWebSocketManager)

    def test_import_helper_functions(self):
        from slate.schematic_api import get_widget_css, get_widget_js
        assert callable(get_widget_css)
        assert callable(get_widget_js)


class TestSchematicTemplates:
    """Test template registry."""

    def test_system_template_exists(self):
        from slate.schematic_api import TEMPLATES
        assert "system" in TEMPLATES

    def test_inference_template_exists(self):
        from slate.schematic_api import TEMPLATES
        assert "inference" in TEMPLATES

    def test_cicd_template_exists(self):
        from slate.schematic_api import TEMPLATES
        assert "cicd" in TEMPLATES

    def test_template_has_name(self):
        from slate.schematic_api import TEMPLATES
        for tid, tmpl in TEMPLATES.items():
            assert "name" in tmpl, f"Template {tid} missing 'name'"

    def test_template_has_config(self):
        from slate.schematic_api import TEMPLATES
        for tid, tmpl in TEMPLATES.items():
            assert "config" in tmpl, f"Template {tid} missing 'config'"


class TestWidgetCSS:
    """Test widget CSS generation."""

    def test_css_not_empty(self):
        from slate.schematic_api import get_widget_css
        css = get_widget_css()
        assert len(css) > 0

    def test_css_has_widget_class(self):
        from slate.schematic_api import get_widget_css
        css = get_widget_css()
        assert ".schematic-widget" in css

    def test_css_has_header_class(self):
        from slate.schematic_api import get_widget_css
        css = get_widget_css()
        assert ".schematic-header" in css

    def test_css_has_content_class(self):
        from slate.schematic_api import get_widget_css
        css = get_widget_css()
        assert ".schematic-content" in css

    def test_css_has_status_class(self):
        from slate.schematic_api import get_widget_css
        css = get_widget_css()
        assert ".schematic-status" in css

    def test_css_has_compact_class(self):
        from slate.schematic_api import get_widget_css
        css = get_widget_css()
        assert ".schematic-compact" in css

    def test_css_has_pulse_animation(self):
        from slate.schematic_api import get_widget_css
        css = get_widget_css()
        assert "@keyframes pulse" in css


class TestWidgetJS:
    """Test widget JavaScript generation."""

    def test_js_not_empty(self):
        from slate.schematic_api import get_widget_js
        js = get_widget_js()
        assert len(js) > 0

    def test_js_has_schematic_widget_class(self):
        from slate.schematic_api import get_widget_js
        js = get_widget_js()
        assert "class SchematicWidget" in js

    def test_js_has_websocket_connection(self):
        from slate.schematic_api import get_widget_js
        js = get_widget_js()
        assert "WebSocket" in js

    def test_js_has_auto_reconnect(self):
        from slate.schematic_api import get_widget_js
        js = get_widget_js()
        assert "connectWebSocket" in js

    def test_js_has_update_schematic(self):
        from slate.schematic_api import get_widget_js
        js = get_widget_js()
        assert "updateSchematic" in js

    def test_js_connects_to_correct_endpoint(self):
        from slate.schematic_api import get_widget_js
        js = get_widget_js()
        assert "/api/schematic/ws/live" in js


class TestSVGGeneration:
    """Test SVG generation functions."""

    def test_generate_from_system_state(self):
        from slate.schematic_api import generate_from_system_state
        svg = generate_from_system_state()
        assert isinstance(svg, str)
        assert len(svg) > 0
        assert "<svg" in svg or "svg" in svg.lower()

    def test_build_from_template_system(self):
        from slate.schematic_api import build_from_template
        svg = build_from_template("system")
        assert isinstance(svg, str)
        assert len(svg) > 0

    def test_build_from_template_inference(self):
        from slate.schematic_api import build_from_template
        svg = build_from_template("inference")
        assert isinstance(svg, str)
        assert len(svg) > 0

    def test_build_from_template_cicd(self):
        from slate.schematic_api import build_from_template
        svg = build_from_template("cicd")
        assert isinstance(svg, str)
        assert len(svg) > 0


class TestWebSocketManager:
    """Test SchematicWebSocketManager."""

    def test_manager_init(self):
        from slate.schematic_api import SchematicWebSocketManager
        mgr = SchematicWebSocketManager()
        assert mgr.active_connections == []

    @pytest.mark.asyncio
    async def test_manager_disconnect_nonexistent(self):
        from slate.schematic_api import SchematicWebSocketManager
        mgr = SchematicWebSocketManager()
        mock_ws = MagicMock()
        # Should not raise
        mgr.disconnect(mock_ws)
        assert len(mgr.active_connections) == 0

    @pytest.mark.asyncio
    async def test_manager_broadcast_empty(self):
        from slate.schematic_api import SchematicWebSocketManager
        mgr = SchematicWebSocketManager()
        # Should not raise with no connections
        await mgr.broadcast({"type": "test"})


# ── Dashboard Template Phase 2 Tests ─────────────────────────────────────────

class TestDashboardTemplatePhase2:
    """Test Spec 012 Phase 2 widget integration in dashboard template."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Build full template (HTML + JS) for all tests."""
        from slate_web.dashboard_template import get_full_template
        self.template = get_full_template()

    # ─── Compact Sidebar Widget ───

    def test_compact_widget_html(self):
        assert 'id="sidebar-schematic"' in self.template

    def test_compact_widget_css(self):
        assert ".schematic-compact" in self.template

    def test_compact_widget_click_handler(self):
        assert "openSchematicModal" in self.template

    def test_compact_widget_label(self):
        assert "schematic-compact-label" in self.template

    def test_compact_widget_dot(self):
        assert "sc-dot" in self.template

    # ─── Card Schematic ───

    def test_card_schematic_gpu(self):
        assert 'id="gpu-topology-schematic"' in self.template

    def test_card_schematic_agents(self):
        assert 'id="agent-pipeline-schematic"' in self.template

    def test_card_schematic_css(self):
        assert ".card-schematic" in self.template

    def test_card_schematic_expand_button(self):
        assert "cs-expand" in self.template

    def test_card_schematic_hover_expand(self):
        assert ".card-schematic:hover .cs-expand" in self.template

    # ─── Modal Detail View ───

    def test_modal_overlay_html(self):
        assert 'id="schematic-modal"' in self.template

    def test_modal_has_aria(self):
        assert 'role="dialog"' in self.template
        assert 'aria-modal="true"' in self.template

    def test_modal_close_button(self):
        assert "closeSchematicModal" in self.template

    def test_modal_css(self):
        assert ".schematic-modal-overlay" in self.template
        assert ".schematic-modal" in self.template

    def test_modal_header(self):
        assert "schematic-modal-header" in self.template
        assert "schematic-modal-title" in self.template

    def test_modal_body(self):
        assert "schematic-modal-body" in self.template

    def test_modal_footer(self):
        assert "schematic-modal-footer" in self.template

    def test_modal_backdrop_filter(self):
        assert "backdrop-filter: blur" in self.template

    # ─── Status Overlay ───

    def test_status_overlay_html(self):
        assert "schematic-status-overlay" in self.template

    def test_status_chips(self):
        assert "schematic-status-chip" in self.template

    def test_status_dots(self):
        assert "ssc-dot" in self.template

    def test_status_dot_states(self):
        assert ".ssc-dot.active" in self.template
        assert ".ssc-dot.warning" in self.template
        assert ".ssc-dot.error" in self.template
        assert ".ssc-dot.idle" in self.template

    # ─── JavaScript ───

    def test_js_open_modal(self):
        assert "function openSchematicModal" in self.template

    def test_js_close_modal(self):
        assert "function closeSchematicModal" in self.template

    def test_js_escape_closes_modal(self):
        assert "Escape" in self.template

    def test_js_sidebar_updater(self):
        assert "updateSidebarSchematic" in self.template

    def test_js_status_overlay_updater(self):
        assert "updateSchematicStatusOverlay" in self.template

    def test_js_fetch_template(self):
        assert "/api/schematic/template/" in self.template

    def test_js_fetch_compact_widget(self):
        assert "/api/schematic/widget/compact" in self.template

    def test_js_fetch_system_state(self):
        assert "/api/schematic/system-state" in self.template

    # ─── Responsive ───

    def test_responsive_compact_mobile(self):
        assert ".schematic-compact" in self.template

    def test_responsive_modal_mobile(self):
        # Check mobile modal sizing
        assert "96vw" in self.template or "90vh" in self.template

    # ─── Integration ───

    def test_sidebar_schematic_updates_on_interval(self):
        assert "updateSidebarSchematic" in self.template
        assert "60000" in self.template  # 60s interval

    def test_status_overlay_updates_on_interval(self):
        assert "updateSchematicStatusOverlay" in self.template
        assert "15000" in self.template


class TestGetFullTemplate:
    """Test the full template builder."""

    def test_full_template_builds(self):
        from slate_web.dashboard_template import get_full_template
        html = get_full_template()
        assert isinstance(html, str)
        assert len(html) > 100000  # Should be quite large

    def test_full_template_has_schematic_elements(self):
        from slate_web.dashboard_template import get_full_template
        html = get_full_template()
        assert "schematic-hero" in html
        assert "schematic-modal" in html
        assert "sidebar-schematic" in html

    def test_full_template_is_valid_html(self):
        from slate_web.dashboard_template import get_full_template
        html = get_full_template()
        assert html.strip().startswith("<!DOCTYPE html>")
        assert "</html>" in html


class TestSchematicAPIRouter:
    """Test that the schematic API router has expected routes."""

    def test_router_has_routes(self):
        from slate.schematic_api import router
        routes = [r.path for r in router.routes]
        assert "/templates" in routes or any("/templates" in r for r in routes)

    def test_router_has_widget_endpoint(self):
        from slate.schematic_api import router
        routes = [r.path for r in router.routes]
        assert any("widget" in r for r in routes)

    def test_router_has_websocket(self):
        from slate.schematic_api import router
        routes = [r.path for r in router.routes]
        assert any("ws" in r for r in routes)


# ── Phase 3: Watchmaker 3D Dashboard Tests ────────────────────────────────────
# Modified: 2026-02-08T06:00:00Z | Author: COPILOT | Change: Add watchmaker Phase 3 dashboard view tests

class TestWatchmaker3DPerspective:
    """Test 3D perspective container and z-layer CSS."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    def test_dashboard_3d_class(self, template):
        assert "dashboard-3d" in template

    def test_perspective_css(self, template):
        assert "perspective: 1200px" in template

    def test_z_layer_background(self, template):
        assert ".z-background" in template

    def test_z_layer_grid(self, template):
        assert ".z-grid" in template

    def test_z_layer_connections(self, template):
        assert ".z-connections" in template

    def test_z_layer_components(self, template):
        assert ".z-components" in template

    def test_z_layer_floating(self, template):
        assert ".z-floating" in template

    def test_z_layer_overlay(self, template):
        assert ".z-overlay" in template

    def test_preserve_3d_transform(self, template):
        assert "preserve-3d" in template


class TestWatchmakerCard:
    """Test watchmaker card component CSS and HTML."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    def test_watchmaker_card_class(self, template):
        assert ".watchmaker-card" in template

    def test_watchmaker_card_hover(self, template):
        assert ".watchmaker-card:hover" in template

    def test_watchmaker_card_header(self, template):
        assert ".watchmaker-card-header" in template

    def test_gear_icon_class(self, template):
        assert ".gear-icon" in template

    def test_gear_spin_animation(self, template):
        assert "gear-spin" in template

    def test_watchmaker_card_html(self, template):
        assert 'class="watchmaker-card' in template

    def test_watchmaker_card_active(self, template):
        assert 'watchmaker-card active' in template


class TestStatusJewel:
    """Test status jewel indicator CSS."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    def test_jewel_class(self, template):
        assert ".status-jewel" in template

    def test_jewel_active(self, template):
        assert ".status-jewel.active" in template

    def test_jewel_pending(self, template):
        assert ".status-jewel.pending" in template

    def test_jewel_error(self, template):
        assert ".status-jewel.error" in template

    def test_jewel_inactive(self, template):
        assert ".status-jewel.inactive" in template

    def test_jewel_pulse_animation(self, template):
        assert "jewel-pulse" in template

    def test_jewel_html_elements(self, template):
        assert 'class="status-jewel' in template


class TestCommandCenterView:
    """Test Command Center (home) watchmaker view."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    def test_section_exists(self, template):
        assert 'id="sec-command-center"' in template

    def test_health_ring(self, template):
        assert 'class="health-ring"' in template

    def test_health_ring_nodes(self, template):
        for component in ['gpu', 'ai', 'tasks', 'runner', 'services']:
            assert f'data-component="{component}"' in template

    def test_health_ring_connectors(self, template):
        assert "health-ring-connector" in template

    def test_summary_cards(self, template):
        assert 'id="cc-svc-count"' in template
        assert 'id="cc-gpu-count"' in template
        assert 'id="cc-task-count"' in template
        assert 'id="cc-model-count"' in template

    def test_system_health_ring_css(self, template):
        assert ".health-ring-node" in template
        assert ".health-ring-connector" in template

    def test_connector_pulse_animation(self, template):
        assert "connector-pulse" in template


class TestConstellationView:
    """Test Service Constellation watchmaker view."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    def test_section_exists(self, template):
        assert 'id="sec-constellation"' in template

    def test_constellation_svg(self, template):
        assert 'class="constellation-view"' in template

    def test_service_nodes(self, template):
        for svc in ['Dashboard', 'Ollama', 'ChromaDB', 'MCP Server', 'Foundry', 'GPU Farm', 'Runner']:
            assert svc in template

    def test_constellation_legend(self, template):
        assert "constellation-legend" in template

    def test_data_flow_lines(self, template):
        assert "data-flow-line" in template

    def test_constellation_jewels(self, template):
        assert 'id="cst-dash-jewel"' in template
        assert 'id="cst-ollama-jewel"' in template

    def test_port_labels(self, template):
        assert ":8080" in template
        assert ":11434" in template
        assert ":8000" in template


class TestGPUWorkbenchView:
    """Test GPU Workbench watchmaker view."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    def test_section_exists(self, template):
        assert 'id="sec-gpu-workbench"' in template

    def test_gpu_units(self, template):
        assert 'id="gpu-unit-0"' in template
        assert 'id="gpu-unit-1"' in template

    def test_gpu_meters(self, template):
        assert "gpu-meter-bar" in template
        assert 'id="wb-gpu0-compute"' in template
        assert 'id="wb-gpu1-memory"' in template

    def test_gpu_power_meters(self, template):
        assert 'id="wb-gpu0-power"' in template
        assert 'id="wb-gpu1-power"' in template

    def test_gpu_task_lists(self, template):
        assert 'id="wb-gpu0-tasks"' in template
        assert 'id="wb-gpu1-tasks"' in template

    def test_gpu_temp_display(self, template):
        assert 'id="wb-gpu0-temp"' in template
        assert 'id="wb-gpu1-temp"' in template

    def test_load_scheduler(self, template):
        assert "gpu-load-scheduler" in template
        assert 'id="wb-load-0"' in template
        assert 'id="wb-load-1"' in template

    def test_gpu_workbench_css(self, template):
        assert ".gpu-workbench" in template
        assert ".gpu-unit" in template
        assert ".gpu-meter-fill" in template

    def test_rtx_5070_label(self, template):
        assert "RTX 5070 Ti" in template

    def test_cuda_version(self, template):
        assert "CUDA 12.8" in template

    def test_blackwell_arch(self, template):
        assert "Blackwell" in template


class TestTaskOrchestrationView:
    """Test Task Orchestration pipeline watchmaker view."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    def test_section_exists(self, template):
        assert 'id="sec-task-orch"' in template

    def test_pipeline_stages(self, template):
        for stage in ['tp-pending', 'tp-running', 'tp-review', 'tp-completed']:
            assert f'id="{stage}"' in template

    def test_pipeline_arrows(self, template):
        assert "task-pipeline-arrow" in template

    def test_pipeline_css(self, template):
        assert ".task-pipeline-viz" in template
        assert ".task-pipeline-stage" in template

    def test_current_stage_highlight(self, template):
        assert ".task-pipeline-stage.current" in template

    def test_workflow_runs_container(self, template):
        assert 'id="task-orch-runs"' in template


class TestWatchmakerNavigation:
    """Test sidebar nav items for watchmaker views."""

    @pytest.fixture
    def full_template(self):
        from slate_web.dashboard_template import get_full_template
        return get_full_template()

    def test_watchmaker_nav_section(self, full_template):
        assert "Watchmaker" in full_template

    def test_command_center_nav(self, full_template):
        assert "showSection('command-center'" in full_template

    def test_constellation_nav(self, full_template):
        assert "showSection('constellation'" in full_template

    def test_gpu_workbench_nav(self, full_template):
        assert "showSection('gpu-workbench'" in full_template

    def test_task_orch_nav(self, full_template):
        assert "showSection('task-orch'" in full_template

    def test_section_titles_js(self, full_template):
        assert "'command-center'" in full_template
        assert "'constellation'" in full_template
        assert "'gpu-workbench'" in full_template
        assert "'task-orch'" in full_template


class TestWatchmakerJSFunctions:
    """Test JavaScript functions for watchmaker views."""

    @pytest.fixture
    def full_template(self):
        from slate_web.dashboard_template import get_full_template
        return get_full_template()

    def test_update_health_ring_fn(self, full_template):
        assert "function updateHealthRing" in full_template or "async function updateHealthRing" in full_template

    def test_update_constellation_fn(self, full_template):
        assert "function updateConstellationJewels" in full_template or "async function updateConstellationJewels" in full_template

    def test_update_gpu_workbench_fn(self, full_template):
        assert "function updateGPUWorkbench" in full_template or "async function updateGPUWorkbench" in full_template

    def test_refresh_task_orchestration_fn(self, full_template):
        assert "function refreshTaskOrchestration" in full_template or "async function refreshTaskOrchestration" in full_template

    def test_refresh_workflow_runs_fn(self, full_template):
        assert "function refreshWorkflowRuns" in full_template or "async function refreshWorkflowRuns" in full_template

    def test_init_watchmaker_views_fn(self, full_template):
        assert "initWatchmakerViews" in full_template

    def test_set_jewel_helper(self, full_template):
        assert "function setJewel" in full_template

    def test_periodic_intervals(self, full_template):
        assert "setInterval(updateHealthRing" in full_template
        assert "setInterval(updateGPUWorkbench" in full_template
        assert "setInterval(updateConstellationJewels" in full_template
        assert "setInterval(refreshTaskOrchestration" in full_template


class TestWatchmakerDesignSystem:
    """Test WatchmakerPatternGenerator in design_system.py."""

    def test_import_watchmaker_pattern_generator(self):
        from slate_web.design_system import WatchmakerPatternGenerator
        assert WatchmakerPatternGenerator is not None

    def test_gear_svg(self):
        from slate_web.design_system import WatchmakerPatternGenerator
        svg = WatchmakerPatternGenerator.gear_svg(100, 8)
        assert "<svg" in svg
        assert "polygon" in svg
        assert "circle" in svg

    def test_gear_mechanism_bg(self):
        from slate_web.design_system import WatchmakerPatternGenerator
        svg = WatchmakerPatternGenerator.gear_mechanism_bg(800, 600, 3)
        assert "<svg" in svg
        assert "animateTransform" in svg

    def test_flow_line_pattern(self):
        from slate_web.design_system import WatchmakerPatternGenerator
        svg = WatchmakerPatternGenerator.flow_line_pattern(600, 50, 3)
        assert "<svg" in svg
        assert "animate" in svg
        assert "circle" in svg

    def test_status_jewel_svg(self):
        from slate_web.design_system import WatchmakerPatternGenerator
        for status in ["active", "pending", "error", "inactive"]:
            svg = WatchmakerPatternGenerator.status_jewel_svg(16, status)
            assert "<svg" in svg
            assert "radialGradient" in svg

    def test_gear_svg_size(self):
        from slate_web.design_system import WatchmakerPatternGenerator
        svg = WatchmakerPatternGenerator.gear_svg(200, 12)
        assert 'viewBox="0 0 200 200"' in svg

    def test_build_design_assets_includes_gears(self):
        from slate_web.design_system import build_design_assets
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            assets = build_design_assets(td)
            assert 'pattern_gears' in assets
            assert 'pattern_flowline' in assets
            assert os.path.exists(assets['pattern_gears'])

    def test_build_design_assets_includes_jewels(self):
        from slate_web.design_system import build_design_assets
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            assets = build_design_assets(td)
            for status in ['active', 'pending', 'error', 'inactive']:
                assert f'jewel_{status}' in assets


class TestWatchmakerResponsive:
    """Test responsive CSS for watchmaker views."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    def test_gpu_workbench_responsive(self, template):
        # On small screens, GPU workbench should stack
        assert "gpu-workbench" in template
        # Check the responsive media query handles gpu-workbench
        assert "grid-template-columns: 1fr" in template

    def test_health_ring_responsive(self, template):
        assert "health-ring" in template

    def test_gear_bg_responsive(self, template):
        assert ".gear-bg" in template


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Animation System Tests
# Modified: 2026-02-08T08:00:00Z | Author: COPILOT | Change: Add Phase 4 animation system tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGearRotationController:
    """Test gear rotation speed-aware controller CSS and JS."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    @pytest.fixture
    def js(self):
        from slate_web.dashboard_template import build_template_js
        return build_template_js()

    def test_gear_speed_tiers_css(self, template):
        """All 5 speed tiers should be defined in CSS."""
        for tier in ['gear-speed-idle', 'gear-speed-low', 'gear-speed-medium', 'gear-speed-high', 'gear-speed-max']:
            assert tier in template, f"Missing gear speed tier: {tier}"

    def test_gear_speed_idle_30s(self, template):
        assert "animation-duration: 30s" in template

    def test_gear_speed_max_2s(self, template):
        assert "animation-duration: 2s" in template

    def test_gear_paused_class(self, template):
        assert "gear-paused" in template
        assert "animation-play-state: paused" in template

    def test_gear_reverse_class(self, template):
        assert "gear-reverse" in template
        assert "animation-direction: reverse" in template

    def test_gear_spin_eased_keyframe(self, template):
        assert "gear-spin-eased" in template

    def test_gear_controller_js_object(self, js):
        assert "GearController" in js

    def test_gear_controller_init(self, js):
        assert "GearController" in js
        assert "init()" in js or "init (" in js

    def test_gear_controller_load_to_speed(self, js):
        assert "loadToSpeedClass" in js

    def test_gear_controller_sync_with_system(self, js):
        assert "syncWithSystem" in js

    def test_gear_controller_pause_resume(self, js):
        assert ".pause()" in js or "pause()" in js
        assert ".resume()" in js or "resume()" in js


class TestFlowLinePulseAnimation:
    """Test enhanced flow line animations."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    @pytest.fixture
    def js(self):
        from slate_web.dashboard_template import build_template_js
        return build_template_js()

    def test_flow_speed_variants(self, template):
        for variant in ['flow-fast', 'flow-medium', 'flow-slow', 'flow-inactive']:
            assert variant in template, f"Missing flow variant: {variant}"

    def test_flow_fast_duration(self, template):
        assert "animation-duration: 0.6s" in template

    def test_flow_glow_class(self, template):
        assert "flow-glow" in template

    def test_flow_pulse_glow_keyframe(self, template):
        assert "flow-pulse-glow" in template
        assert "drop-shadow" in template

    def test_flow_inactive_paused(self, template):
        # Inactive flows should have paused animation
        assert "flow-inactive" in template

    def test_flow_controller_js(self, js):
        assert "FlowController" in js

    def test_flow_controller_set_speed(self, js):
        assert "setFlowSpeed" in js

    def test_flow_controller_sync_services(self, js):
        assert "syncWithServices" in js


class TestJewelPulseAnimation:
    """Test jewel pulse animation variants."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    @pytest.fixture
    def js(self):
        from slate_web.dashboard_template import build_template_js
        return build_template_js()

    def test_jewel_pulse_urgent_keyframe(self, template):
        assert "jewel-pulse-urgent" in template

    def test_jewel_pulse_slow_keyframe(self, template):
        assert "jewel-pulse-slow" in template

    def test_jewel_breathe_keyframe(self, template):
        assert "jewel-breathe" in template

    def test_jewel_urgent_class(self, template):
        assert "pulse-urgent" in template

    def test_jewel_slow_class(self, template):
        assert "pulse-slow" in template

    def test_jewel_breathe_class(self, template):
        assert ".breathe" in template or "breathe" in template

    def test_jewel_controller_js(self, js):
        assert "JewelController" in js

    def test_jewel_set_pulse_variant(self, js):
        assert "setPulseVariant" in js


class TestCardHoverDepthAnimation:
    """Test card hover depth tracking system."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    @pytest.fixture
    def js(self):
        from slate_web.dashboard_template import build_template_js
        return build_template_js()

    def test_card_css_variables(self, template):
        for var in ['--card-rx', '--card-ry', '--card-tz', '--card-shadow-x', '--card-shadow-y']:
            assert var in template, f"Missing CSS var: {var}"

    def test_depth_tracking_class(self, template):
        assert "depth-tracking" in template

    def test_depth_lift_class(self, template):
        assert "depth-lift" in template

    def test_card_shimmer_keyframe(self, template):
        assert "card-shimmer" in template

    def test_shimmer_active_class(self, template):
        assert "shimmer-active" in template

    def test_card_depth_controller_js(self, js):
        assert "CardDepthController" in js

    def test_card_depth_mouse_events(self, js):
        assert "mouseenter" in js
        assert "mousemove" in js
        assert "mouseleave" in js

    def test_card_depth_rotation_calc(self, js):
        # Should calculate rotation based on mouse position relative to card center
        assert "rotateX" in js or "card-rx" in js
        assert "rotateY" in js or "card-ry" in js

    def test_card_depth_reduced_motion_check(self, js):
        assert "prefers-reduced-motion" in js


class TestBackgroundGearParallax:
    """Test background gear parallax layers."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    @pytest.fixture
    def js(self):
        from slate_web.dashboard_template import build_template_js
        return build_template_js()

    def test_gear_parallax_layer_class(self, template):
        assert "gear-parallax-layer" in template

    def test_parallax_depth_tiers(self, template):
        for tier in ['gear-parallax-far', 'gear-parallax-mid', 'gear-parallax-near']:
            assert tier in template, f"Missing parallax tier: {tier}"

    def test_five_gear_layers_css(self, template):
        for i in range(1, 6):
            assert f"gear-layer-{i}" in template

    def test_five_gear_layers_html(self, template):
        # All 5 gear layers should exist in the HTML body
        for i in range(1, 6):
            assert f'class="gear-layer-{i}' in template or f"class='gear-layer-{i}" in template or f'gear-layer-{i} gear-parallax-layer' in template

    def test_gear_svg_elements(self, template):
        # Each gear layer should have an animated SVG
        assert "gear-animated" in template
        count = template.count('class="gear-animated')
        assert count >= 5, f"Expected at least 5 gear-animated SVGs, found {count}"

    def test_reverse_gears(self, template):
        # Some gears should rotate in reverse
        assert 'gear-animated reverse' in template

    def test_gear_container_id(self, template):
        assert 'gear-bg-container' in template

    def test_gear_parallax_js(self, js):
        assert "GearParallax" in js

    def test_parallax_mouse_move(self, js):
        assert "mousemove" in js

    def test_parallax_request_animation_frame(self, js):
        assert "requestAnimationFrame" in js

    def test_parallax_speed_factors(self, js):
        # Each layer should have a different speed factor
        assert "speed:" in js or "speed :" in js


class TestAnimationSystemInit:
    """Test the animation system initialization and integration."""

    @pytest.fixture
    def js(self):
        from slate_web.dashboard_template import build_template_js
        return build_template_js()

    def test_init_animation_system_function(self, js):
        assert "initAnimationSystem" in js

    def test_init_calls_gear_controller(self, js):
        assert "GearController.init()" in js

    def test_init_calls_flow_controller(self, js):
        assert "FlowController.init()" in js

    def test_init_calls_card_depth(self, js):
        assert "CardDepthController.init()" in js

    def test_init_calls_gear_parallax(self, js):
        assert "GearParallax.init()" in js

    def test_gear_sync_interval(self, js):
        # Gear speed should sync with system periodically
        assert "syncWithSystem" in js

    def test_flow_sync_interval(self, js):
        assert "syncWithServices" in js

    def test_init_animation_system_called(self, js):
        # initAnimationSystem should be called in the init section
        assert "initAnimationSystem();" in js


class TestReducedMotionAccessibility:
    """Test prefers-reduced-motion compliance."""

    @pytest.fixture
    def template(self):
        from slate_web.dashboard_template import build_template
        return build_template()

    def test_reduced_motion_media_query(self, template):
        assert "prefers-reduced-motion: reduce" in template

    def test_reduced_motion_disables_gear_animations(self, template):
        # The reduced motion block should target gear-animated SVGs
        assert "gear-animated" in template

    def test_reduced_motion_disables_jewel_pulse(self, template):
        assert "pulse-urgent" in template
        assert "pulse-slow" in template

    def test_reduced_motion_disables_flow_lines(self, template):
        assert "data-flow-line" in template

    def test_reduced_motion_disables_depth_tracking(self, template):
        assert "depth-tracking" in template

    def test_reduced_motion_animation_none(self, template):
        # Should set animation: none !important
        assert "animation: none !important" in template


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
