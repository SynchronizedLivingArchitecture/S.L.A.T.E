# Spec 011: Schematic Diagram SDK - Implementation Tasks

## Phase 1: Core Engine
- [x] Create specification document
- [ ] Create `slate/schematic_sdk/` directory structure
- [ ] Implement `components.py` - Component dataclasses
- [ ] Implement `theme.py` - Theme integration with design_tokens
- [ ] Implement `engine.py` - SchematicEngine core class

## Phase 2: SVG Renderer
- [ ] Implement `svg_renderer.py` - SVG generation pipeline
- [ ] Add defs generation (gradients, filters, markers)
- [ ] Add component rendering (nodes, shapes)
- [ ] Add connection rendering (lines, arrows)
- [ ] Add grid overlay and background
- [ ] Add legend and version badge

## Phase 3: Layout Algorithms
- [ ] Implement `layout.py` - Layout base class
- [ ] Implement HierarchicalLayout algorithm
- [ ] Implement ForceDirectedLayout algorithm
- [ ] Implement GridLayout algorithm

## Phase 4: Integration
- [ ] Implement `cli.py` - CLI entry point
- [ ] Implement `library.py` - Pre-built component library
- [ ] Implement `exporters.py` - Export handlers
- [ ] Create `.claude/commands/slate-schematic.md`
- [ ] Add MCP tool to `slate/mcp_server.py`
- [ ] Update `slate/slate_generative_ui.py`

## Phase 5: Testing & Documentation
- [ ] Create `tests/test_schematic_sdk.py`
- [ ] Generate sample diagram `docs/assets/slate-system-schematic.svg`
- [ ] Test CLI commands
- [ ] Test MCP tool integration
- [ ] Validate GitHub Pages rendering

## Files to Create
- `slate/schematic_sdk/__init__.py`
- `slate/schematic_sdk/engine.py`
- `slate/schematic_sdk/components.py`
- `slate/schematic_sdk/layout.py`
- `slate/schematic_sdk/theme.py`
- `slate/schematic_sdk/svg_renderer.py`
- `slate/schematic_sdk/library.py`
- `slate/schematic_sdk/exporters.py`
- `slate/schematic_sdk/cli.py`
- `.claude/commands/slate-schematic.md`

## Files to Modify
- `slate/mcp_server.py` - Add slate_schematic tool
- `slate/slate_generative_ui.py` - Add SchematicGenerator class
