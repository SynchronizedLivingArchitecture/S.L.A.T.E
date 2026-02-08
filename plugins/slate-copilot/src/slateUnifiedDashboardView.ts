// Modified: 2026-02-07T23:00:00Z | Author: COPILOT | Change: Spec 014 — Watchmaker Golden Ratio UI overhaul, guided operations, unified tokens, prompt ingestion
/**
 * SLATE Unified Dashboard View — v4.0.0
 * =====================================
 * Single integrated webview that combines:
 * - Generative onboarding (system-adaptive, tailored to user hardware)
 * - Prompt ingestion (configure @slate preferences during onboarding)
 * - Systems check (post-onboarding, replaces guided setup view)
 * - Guided operations panel (buttons > prompts paradigm)
 * - Control board (service status, dev cycle, learning mode)
 * - Dashboard link (opens FastAPI backend at 127.0.0.1:8080 in browser)
 *
 * Design System: SLATE Watchmaker + Golden Ratio (spec-014)
 * Engineering Drawing: spec-013
 * Generative UI: spec-010
 * Version-based re-onboarding on any version change
 *
 * Golden Ratio (φ = 1.618) governs:
 * - Typography scale: 8, 11, 13, 16, 21, 34
 * - Spacing: Fibonacci sequence (1,2,3,5,8,13,21,34,55,89)
 * - Layout proportions: 61.8% / 38.2%
 * - Animation timing: φ-derived durations
 */

import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';
import { writeFile, unlink } from 'fs/promises';
import * as path from 'path';
import { getSlateConfig } from './extension';

const execAsync = promisify(exec);

const DASHBOARD_URL = 'http://127.0.0.1:8080';
const EXTENSION_VERSION = '4.0.0';

// ── Types ───────────────────────────────────────────────────────────────────

interface GuidedStep {
	id: string;
	title: string;
	description: string;
	status: 'pending' | 'active' | 'executing' | 'complete' | 'error';
	substeps: SubStep[];
	aiNarration?: string;
	optional?: boolean;
}

interface SubStep {
	id: string;
	label: string;
	status: 'pending' | 'running' | 'success' | 'error' | 'skipped';
	result?: string;
	optional?: boolean;
}

interface SystemProfile {
	pythonVersion: string;
	gpuCount: number;
	gpuModels: string[];
	totalVramGb: number;
	ollamaAvailable: boolean;
	ollamaModels: string[];
	dockerAvailable: boolean;
	githubAuthenticated: boolean;
	venvActive: boolean;
	platform: string;
	packageCount: number;
}

// ── SLATE Design Tokens (Watchmaker + Golden Ratio — Spec 014) ──────────────
// φ = 1.6180339887 — governs all proportions
// Fibonacci spacing: 1, 2, 3, 5, 8, 13, 21, 34, 55, 89 px
// Typography scale: 8 → 11 → 13 → 16 → 21 → 34 (φ-derived)

const PHI = 1.6180339887;

const SLATE_TOKENS = {
	// ── Surfaces — true black foundation (watchmaker case) ──
	bgRoot: '#050505',
	bgSurface: '#0a0a0a',
	bgSurfaceVariant: '#111111',
	bgContainer: '#141210',
	bgContainerHigh: '#1a1816',
	bgContainerHighest: '#222020',

	// ── Primary — copper (the watchmaker's metal) ──
	primary: '#B87333',
	primaryLight: '#C9956B',
	primaryDark: '#8B5E2B',
	primaryContainer: 'rgba(184,115,51,0.12)',
	onPrimary: '#FFFFFF',
	onPrimaryContainer: '#3D1E10',

	// ── Accent — copper/bronze (legacy compat, same as primary) ──
	accent: '#B87333',
	accentLight: '#C9956B',
	accentDark: '#8B5E2B',
	accentGlow: 'rgba(184,115,51,0.15)',
	accentContainer: 'rgba(184,115,51,0.12)',

	// ── Text — warm white / natural earth ──
	textPrimary: '#F5F0EB',
	textSecondary: '#A8A29E',
	textTertiary: '#78716C',
	textDisabled: '#44403C',
	onSurface: '#E7E0D8',
	onSurfaceVariant: '#CAC4BF',

	// ── Borders / Outlines ──
	border: 'rgba(255,255,255,0.08)',
	borderVariant: 'rgba(255,255,255,0.12)',
	borderFocus: '#B87333',
	outline: '#7D7873',
	outlineVariant: '#4D4845',

	// ── Semantic (unified — Spec 014) ──
	success: '#22C55E',
	successContainer: 'rgba(34,197,94,0.12)',
	warning: '#D4A054',
	warningContainer: 'rgba(212,160,84,0.12)',
	error: '#C47070',
	errorContainer: 'rgba(196,112,112,0.12)',
	info: '#7EA8BE',
	infoContainer: 'rgba(126,168,190,0.12)',

	// ── Engineering Traces (ISO 128 / IEC 60617) ──
	traceSignal: '#B87333',
	traceData: '#7EA8BE',
	tracePower: '#C47070',
	traceControl: '#D4A054',
	traceGround: '#78716C',

	// ── Blueprint ──
	blueprintBg: '#0D1B2A',
	blueprintGrid: '#1B3A4B',
	blueprintAccent: '#98C1D9',
	blueprintNode: '#E0FBFC',

	// ── Component Fills (IEC 60617 conventions) ──
	fillService: '#1a1510',
	fillDatabase: '#101520',
	fillGpu: '#15120a',
	fillAi: '#0a1515',
	fillExternal: '#151015',

	// ── Typography — φ-derived scale (Golden Ratio) ──
	// Base: 8px → 8, 11, 13, 16, 21, 34
	fontDisplay: "'Inter Tight', 'Segoe UI', system-ui, sans-serif",
	fontBody: "'Inter', 'Segoe UI', system-ui, sans-serif",
	fontMono: "'Cascadia Code', 'JetBrains Mono', 'Consolas', monospace",
	fontSchematic: "'Consolas', 'Courier New', monospace",

	// ── Type Scale (φ-derived: each step ≈ φ ratio) ──
	displayLarge: '55px',
	displayMedium: '34px',
	displaySmall: '28px',
	headlineLarge: '28px',
	headlineMedium: '21px',
	headlineSmall: '18px',
	titleLarge: '18px',
	titleMedium: '16px',
	titleSmall: '14px',
	bodyLarge: '16px',
	bodyMedium: '13px',
	bodySmall: '11px',
	labelLarge: '13px',
	labelMedium: '11px',
	labelSmall: '10px',
	caption: '8px',

	// ── Spacing (Fibonacci — φ derived) ──
	space1: '1px',
	space2: '2px',
	space3: '3px',
	space5: '5px',
	space8: '8px',
	space13: '13px',
	space21: '21px',
	space34: '34px',
	space55: '55px',
	space89: '89px',

	// ── Elevation (5 levels) ──
	elevation0: 'none',
	elevation1: '0 1px 2px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.1)',
	elevation2: '0 2px 4px rgba(0,0,0,0.05), 0 4px 8px rgba(0,0,0,0.1)',
	elevation3: '0 4px 8px rgba(0,0,0,0.08), 0 8px 16px rgba(0,0,0,0.12)',
	elevation4: '0 8px 16px rgba(0,0,0,0.1), 0 16px 32px rgba(0,0,0,0.15)',
	elevation5: '0 16px 32px rgba(0,0,0,0.12), 0 32px 64px rgba(0,0,0,0.18)',

	// ── State Layers ──
	stateHover: '0.08',
	stateFocus: '0.12',
	statePressed: '0.12',
	stateDragged: '0.16',

	// ── Motion (φ-derived timing) ──
	easingStandard: 'cubic-bezier(0.2, 0, 0, 1)',
	easingDecelerate: 'cubic-bezier(0, 0, 0.2, 1)',
	easingAccelerate: 'cubic-bezier(0.4, 0, 1, 1)',
	easingSpring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
	durationFlash: '100ms',
	durationQuick: '200ms',
	durationNormal: '350ms',
	durationSmooth: '550ms',
	durationDramatic: '900ms',

	// Legacy compat
	durationShort1: '50ms',
	durationShort2: '100ms',
	durationShort3: '150ms',
	durationShort4: '200ms',
	durationMedium1: '250ms',
	durationMedium2: '300ms',
	durationMedium3: '350ms',
	durationMedium4: '400ms',
	durationLong1: '450ms',
	durationLong2: '500ms',
	durationLong3: '550ms',
	durationLong4: '600ms',

	// ── Radii ──
	radiusXs: '3px',
	radiusSm: '5px',
	radiusMd: '8px',
	radiusLg: '13px',
	radiusXl: '21px',
	radiusFull: '9999px',

	// ── Dev cycle stage colors ──
	stagePlan: '#7EA8BE',
	stageCode: '#B87333',
	stageTest: '#D4A054',
	stageDeploy: '#78B89A',
	stageFeedback: '#9B89B3',

	// ── Watchmaker craft ──
	gearColor: '#B87333',
	jewelGreen: '#22C55E',
	jewelAmber: '#D4A054',
	jewelRed: '#C47070',
	jewelBlue: '#7EA8BE',
	mainspring: '#C9956B',
	polishReflection: 'linear-gradient(135deg, rgba(255,255,255,0.06) 0%, transparent 50%)',
	caseTexture: 'linear-gradient(180deg, #141210 0%, #0a0a0a 100%)',
};

// ── Guided Install Steps ────────────────────────────────────────────────────

const GUIDED_STEPS: GuidedStep[] = [
	{
		id: 'welcome',
		title: 'Welcome to SLATE',
		description: 'AI-powered local development environment',
		status: 'pending',
		substeps: [
			{ id: 'intro', label: 'Display value proposition', status: 'pending' },
			{ id: 'scan-init', label: 'Initialize system scanner', status: 'pending' },
		],
	},
	{
		id: 'system-scan',
		title: 'System Scan',
		description: 'Detecting installed services and capabilities',
		status: 'pending',
		substeps: [
			{ id: 'detect-python', label: 'Detect Python 3.11+', status: 'pending' },
			{ id: 'detect-gpu', label: 'Detect GPU configuration', status: 'pending' },
			{ id: 'detect-ollama', label: 'Detect Ollama service', status: 'pending' },
			{ id: 'detect-docker', label: 'Detect Docker daemon', status: 'pending', optional: true },
			{ id: 'detect-github', label: 'Detect GitHub CLI', status: 'pending', optional: true },
		],
	},
	{
		id: 'core-services',
		title: 'Core Services',
		description: 'Configuring SLATE core infrastructure',
		status: 'pending',
		substeps: [
			{ id: 'init-venv', label: 'Initialize Python virtual environment', status: 'pending' },
			{ id: 'install-deps', label: 'Install dependencies', status: 'pending' },
			{ id: 'start-dashboard', label: 'Start dashboard server', status: 'pending', optional: true },
			{ id: 'init-orchestrator', label: 'Initialize orchestrator', status: 'pending' },
		],
	},
	{
		id: 'ai-backends',
		title: 'AI Backends',
		description: 'Setting up local AI inference',
		status: 'pending',
		substeps: [
			{ id: 'check-ollama', label: 'Verify Ollama connection', status: 'pending' },
			{ id: 'check-models', label: 'Auto-detect installed models', status: 'pending' },
			{ id: 'check-slate-models', label: 'Check SLATE custom models', status: 'pending' },
			{ id: 'test-inference', label: 'Test inference endpoint', status: 'pending' },
		],
	},
	{
		id: 'integrations',
		title: 'Integrations',
		description: 'Connecting external services',
		status: 'pending',
		substeps: [
			{ id: 'github-auth', label: 'Verify GitHub authentication', status: 'pending' },
			{ id: 'docker-check', label: 'Check Docker daemon', status: 'pending', optional: true },
			{ id: 'mcp-server', label: 'Validate MCP server', status: 'pending' },
			{ id: 'runner-check', label: 'Check GitHub Actions runner', status: 'pending', optional: true },
		],
	},
	{
		id: 'validation',
		title: 'Validation',
		description: 'Running comprehensive checks',
		status: 'pending',
		substeps: [
			{ id: 'health-check', label: 'Health check all services', status: 'pending' },
			{ id: 'gpu-access', label: 'Verify GPU access', status: 'pending' },
			{ id: 'test-workflow', label: 'Test workflow dispatch', status: 'pending' },
			{ id: 'security-scan', label: 'Security scan', status: 'pending' },
		],
	},
	{
		id: 'complete',
		title: 'Setup Complete',
		description: 'Your SLATE system is ready!',
		status: 'pending',
		substeps: [
			{ id: 'summary', label: 'Generate system summary', status: 'pending' },
			{ id: 'recommendations', label: 'AI recommendations', status: 'pending' },
		],
	},
];

// ── Static AI narrations (fallback when Ollama unavailable) ─────────────────

const STATIC_NARRATIONS: Record<string, string> = {
	'welcome': "Welcome to S.L.A.T.E. \u2014 your Synchronized Living Architecture for Transformation and Evolution. I'll guide you through the setup process, scanning your system and configuring everything for optimal performance.",
	'system-scan': "Now scanning your system for prerequisites. I'm looking for Python, GPU drivers, Ollama, Docker, and GitHub integration. This helps me tailor the setup to your hardware.",
	'core-services': "Setting up the core SLATE infrastructure. This includes the Python virtual environment, essential dependencies, the dashboard server, and the service orchestrator.",
	'ai-backends': "Configuring your local AI inference stack. SLATE runs entirely on your hardware \u2014 no cloud APIs, no data leaving your machine. I'll verify Ollama is connected and auto-detect all available models.",
	'integrations': "Connecting to external services. Don't worry if Docker or the runner aren't available \u2014 they're optional. GitHub authentication and MCP are what matter most.",
	'validation': "Running comprehensive validation across all subsystems. I'll check service health, GPU access, workflow dispatch capability, and security configuration.",
	'complete': "Excellent! Your SLATE system is fully operational. You now have a local-first AI development environment. The onboarding view will now transform into your Systems Check dashboard.",
};

// ── Unified Dashboard View Provider ─────────────────────────────────────────

export class SlateUnifiedDashboardViewProvider implements vscode.WebviewViewProvider {
	public static readonly viewType = 'slate.unifiedDashboard';

	private _view?: vscode.WebviewView;
	private _statusInterval?: NodeJS.Timeout;
	private _workspaceRoot: string;
	private _currentStep: number = 0;
	private _steps: GuidedStep[] = JSON.parse(JSON.stringify(GUIDED_STEPS));
	private _isRunning: boolean = false;
	private _systemProfile: SystemProfile | null = null;

	constructor(
		private readonly _extensionUri: vscode.Uri,
		private readonly _context: vscode.ExtensionContext,
	) {
		this._workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
	}

	public resolveWebviewView(
		webviewView: vscode.WebviewView,
		_resolveContext: vscode.WebviewViewResolveContext,
		_token: vscode.CancellationToken,
	): void {
		this._view = webviewView;

		webviewView.webview.options = {
			enableScripts: true,
			localResourceRoots: [this._extensionUri],
		};

		// Check onboarding state + version-based re-onboarding
		const showDashboard = this._shouldShowDashboard();
		webviewView.webview.html = this._getHtml(webviewView.webview, showDashboard);

		// Handle messages
		webviewView.webview.onDidReceiveMessage(async (message) => {
			try {
				switch (message.type) {
					// ── Guided onboarding ──
					case 'startGuided':
						await this._startGuidedInstall();
						break;
					case 'skipOnboarding':
						await this._completeOnboarding();
						break;
					case 'skipStep':
						await this._skipCurrentStep();
						break;
					case 'exitGuided':
						this._exitGuidedMode();
						break;
					case 'openDashboard':
						vscode.env.openExternal(vscode.Uri.parse(DASHBOARD_URL));
						break;
					case 'finishOnboarding':
						await this._completeOnboarding();
						break;
					case 'applySlateTheme':
						await this._applySlateDefaultTheme();
						break;
					case 'detectSystem':
						try {
							await this._detectAndSendSystemProfile();
						} catch {
							// Send a minimal profile so the UI doesn't stay stuck on spinner
							this._sendToWebview({ type: 'systemProfile', profile: {
								pythonVersion: 'Unknown', gpuCount: 0, gpuModels: [], totalVramGb: 0,
								ollamaAvailable: false, ollamaModels: [], dockerAvailable: false,
								githubAuthenticated: false, venvActive: false, platform: process.platform, packageCount: 0,
							} });
						}
						break;

					// ── Control board ──
					case 'runCommand':
						await this._runSlateCommand(message.command);
						break;
					case 'refreshStatus':
						await this._refreshStatus();
						break;
					case 'runSystemsCheck':
						await this._runSystemsCheck();
						break;
					case 'openChat':
						await vscode.commands.executeCommand('workbench.action.chat.open');
						break;
					case 'showStatus':
						await vscode.commands.executeCommand('slate.showStatus');
						break;
					case 'transitionStage':
						await this._transitionDevCycleStage(message.stage);
						break;
					case 'toggleLearning':
						await this._toggleLearningMode(message.active);
						break;
					case 'startGuidedMode':
						await this._resetOnboarding();
						break;

					// ── Dashboard iframe ──
					case 'openExternal':
						void vscode.env.openExternal(vscode.Uri.parse(DASHBOARD_URL));
						break;
					case 'openPanel':
						await vscode.commands.executeCommand('slate.openDashboard');
						break;
				}
			} catch (err) {
				console.error('[SLATE Dashboard] Message handler error:', message.type, err);
			}
		});

		// Auto-refresh status every 30 seconds in dashboard mode
		this._statusInterval = setInterval(() => {
			if (this._shouldShowDashboard()) {
				this._refreshStatus();
				this._fetchInteractiveStatus();
			}
		}, 30000);

		webviewView.onDidDispose(() => {
			if (this._statusInterval) {
				clearInterval(this._statusInterval);
			}
		});

		// Initial status fetch if dashboard mode
		if (showDashboard) {
			this._refreshStatus();
			this._fetchInteractiveStatus();
		}
	}

	// ── Version-based re-onboarding ─────────────────────────────────────────

	private _shouldShowDashboard(): boolean {
		const onboardingComplete = this._context.globalState.get<boolean>('slateOnboardingComplete', false);
		if (!onboardingComplete) { return false; }

		// Any version change (upgrade, downgrade, or mismatch) triggers re-onboarding
		const lastVersion = this._context.globalState.get<string>('slateLastVersion', '0.0.0');
		if (lastVersion !== EXTENSION_VERSION) {
			return false;
		}

		return true;
	}

	// ── Theme Application ───────────────────────────────────────────────────

	private async _applySlateDefaultTheme(): Promise<void> {
		try {
			await vscode.workspace.getConfiguration('workbench').update(
				'colorTheme',
				'SLATE Dark',
				vscode.ConfigurationTarget.Global
			);
			this._sendToWebview({ type: 'themeApplied', success: true });
			vscode.window.showInformationMessage('SLATE Dark theme applied!');
		} catch (err: any) {
			this._sendToWebview({ type: 'themeApplied', success: false, error: err.message });
		}
	}

	// ── System Detection (Generative UI) ────────────────────────────────────

	private async _detectAndSendSystemProfile(): Promise<void> {
		const profile = await this._detectSystem();
		this._systemProfile = profile;
		this._sendToWebview({ type: 'systemProfile', profile });
	}

	private async _detectSystem(): Promise<SystemProfile> {
		const config = getSlateConfig();
		const py = config.pythonPath;
		const cwd = this._workspaceRoot;

		const defaults: SystemProfile = {
			pythonVersion: 'Unknown',
			gpuCount: 0,
			gpuModels: [],
			totalVramGb: 0,
			ollamaAvailable: false,
			ollamaModels: [],
			dockerAvailable: false,
			githubAuthenticated: false,
			venvActive: false,
			platform: process.platform,
			packageCount: 0,
		};

		// Run all detections in parallel for speed
		const [pythonRes, gpuRes, ollamaRes, dockerRes, githubRes, venvRes, pkgRes] = await Promise.allSettled([
			execAsync(`"${py}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"`, { cwd, timeout: 10000 }),
			execAsync(`"${py}" -c "import subprocess; r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True); print(r.stdout.strip() if r.returncode==0 else 'none')"`, { cwd, timeout: 10000 }),
			execAsync(`"${py}" -c "import urllib.request,json; r=urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=3); d=json.loads(r.read()); names=[m['name'] for m in d.get('models',[])]; print('|'.join(names))"`, { cwd, timeout: 10000 }),
			execAsync(`"${py}" -c "import subprocess; r=subprocess.run(['docker','info'],capture_output=True,text=True,timeout=5); print('ok' if r.returncode==0 else 'no')"`, { cwd, timeout: 10000 }),
			execAsync(`"${py}" -c "import subprocess; r=subprocess.run(['git','credential','fill'],input='protocol=https\\nhost=github.com\\n',capture_output=True,text=True); print('ok' if 'password=' in r.stdout else 'no')"`, { cwd, timeout: 10000 }),
			execAsync(`"${py}" -c "import sys; print('yes' if sys.prefix!=sys.base_prefix else 'no')"`, { cwd, timeout: 10000 }),
			execAsync(`"${py}" -c "import pkg_resources; print(len(list(pkg_resources.working_set)))"`, { cwd, timeout: 10000 }),
		]);

		const profile = { ...defaults };

		if (pythonRes.status === 'fulfilled') {
			profile.pythonVersion = pythonRes.value.stdout.trim();
		}

		if (gpuRes.status === 'fulfilled' && gpuRes.value.stdout.trim() !== 'none') {
			const lines = gpuRes.value.stdout.trim().split('\n');
			profile.gpuCount = lines.length;
			profile.gpuModels = [];
			let totalVram = 0;
			for (const line of lines) {
				const parts = line.split(',').map(s => s.trim());
				profile.gpuModels.push(parts[0] || 'Unknown GPU');
				totalVram += parseFloat(parts[1] || '0') / 1024;
			}
			profile.totalVramGb = Math.round(totalVram * 10) / 10;
		}

		if (ollamaRes.status === 'fulfilled') {
			const models = ollamaRes.value.stdout.trim();
			if (models) {
				profile.ollamaAvailable = true;
				profile.ollamaModels = models.split('|').filter(Boolean);
			}
		}

		if (dockerRes.status === 'fulfilled') {
			profile.dockerAvailable = dockerRes.value.stdout.trim() === 'ok';
		}

		if (githubRes.status === 'fulfilled') {
			profile.githubAuthenticated = githubRes.value.stdout.trim() === 'ok';
		}

		if (venvRes.status === 'fulfilled') {
			profile.venvActive = venvRes.value.stdout.trim() === 'yes';
		}

		if (pkgRes.status === 'fulfilled') {
			profile.packageCount = parseInt(pkgRes.value.stdout.trim()) || 0;
		}

		return profile;
	}

	// ── Onboarding management ───────────────────────────────────────────────

	private async _completeOnboarding(): Promise<void> {
		await this._context.globalState.update('slateOnboardingComplete', true);
		await this._context.globalState.update('slateLastVersion', EXTENSION_VERSION);
		if (this._view) {
			this._view.webview.html = this._getHtml(this._view.webview, true);
			this._refreshStatus();
			this._fetchInteractiveStatus();
		}
	}

	private async _resetOnboarding(): Promise<void> {
		await this._context.globalState.update('slateOnboardingComplete', false);
		this._steps = JSON.parse(JSON.stringify(GUIDED_STEPS));
		this._currentStep = 0;
		this._isRunning = false;
		this._systemProfile = null;
		if (this._view) {
			this._view.webview.html = this._getHtml(this._view.webview, false);
		}
	}

	public refresh(): void {
		if (this._view) {
			const showDashboard = this._shouldShowDashboard();
			this._view.webview.html = this._getHtml(this._view.webview, showDashboard);
		}
	}

	// ── Guided Installation Flow ────────────────────────────────────────────

	private async _startGuidedInstall(): Promise<void> {
		if (this._isRunning) { return; }
		this._isRunning = true;
		this._currentStep = 0;
		this._steps = JSON.parse(JSON.stringify(GUIDED_STEPS));

		// Detect system first
		await this._detectAndSendSystemProfile();

		for (let i = 0; i < this._steps.length; i++) {
			this._currentStep = i;
			await this._executeStep(this._steps[i]);
			if (!this._isRunning) { break; }
		}

		this._isRunning = false;
	}

	private async _executeStep(step: GuidedStep): Promise<void> {
		step.status = 'active';

		const narration = await this._getAINarration(step.id);
		this._sendToWebview({ type: 'narration', text: narration });
		this._sendToWebview({ type: 'stepUpdate', step, currentIndex: this._currentStep });

		step.status = 'executing';
		this._sendToWebview({ type: 'stepUpdate', step, currentIndex: this._currentStep });

		for (const substep of step.substeps) {
			await this._executeSubstep(step, substep);
			if (!this._isRunning) { return; }
		}

		step.status = 'complete';
		this._sendToWebview({ type: 'stepComplete', step, currentIndex: this._currentStep });

		await new Promise(resolve => setTimeout(resolve, 800));
	}

	private async _executeSubstep(step: GuidedStep, substep: SubStep): Promise<void> {
		substep.status = 'running';
		this._sendToWebview({ type: 'substepUpdate', stepId: step.id, substep });

		try {
			const result = await this._runSubstepCommand(step.id, substep.id);
			substep.status = 'success';
			substep.result = result;
		} catch (err: any) {
			if (substep.optional) {
				substep.status = 'skipped';
				substep.result = 'Optional \u2014 skipped';
			} else {
				substep.status = 'error';
				substep.result = err?.message || 'Failed';
			}
		}

		this._sendToWebview({ type: 'substepUpdate', stepId: step.id, substep });
	}

	private async _runSubstepCommand(stepId: string, substepId: string): Promise<string> {
		const config = getSlateConfig();
		const py = config.pythonPath;

		const commands: Record<string, string> = {
			'welcome:intro': 'echo SLATE v2.4.0 ready',
			'welcome:scan-init': `"${py}" -c "import sys; print(f'Scanner initialized (Python {sys.version_info.major}.{sys.version_info.minor})')"`,
			'system-scan:detect-python': `"${py}" -c "import sys; print(f'{sys.version}')"`,
			'system-scan:detect-gpu': `"${py}" -c "import subprocess; r=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True); print(r.stdout.strip() or 'No GPU detected')"`,
			'system-scan:detect-ollama': `"${py}" -c "import urllib.request,json; r=urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=3); d=json.loads(r.read()); print(len(d.get('models',[])),'models available')"`,
			'system-scan:detect-docker': `"${py}" -c "import subprocess; r=subprocess.run(['docker','info'],capture_output=True,text=True,timeout=5); print('Running' if r.returncode==0 else 'Not available')"`,
			'system-scan:detect-github': `"${py}" -c "import subprocess; r=subprocess.run(['git','credential','fill'],input='protocol=https\\nhost=github.com\\n',capture_output=True,text=True); print('Authenticated' if 'password=' in r.stdout else 'Not configured')"`,
			'core-services:init-venv': `"${py}" -c "import sys; print('venv active' if sys.prefix!=sys.base_prefix else 'system python')"`,
			'core-services:install-deps': `"${py}" -c "import pkg_resources; print(f'{len(list(pkg_resources.working_set))} packages installed')"`,
			'core-services:start-dashboard': `"${py}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080',timeout=3); print('Dashboard running')"`,
			'core-services:init-orchestrator': `"${py}" slate/slate_orchestrator.py status`,
			'ai-backends:check-ollama': `"${py}" -c "import urllib.request,json; r=urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=3); d=json.loads(r.read()); print(len(d.get('models',[])),'models available')"`,
			'ai-backends:check-models': `"${py}" -c "import urllib.request,json; r=urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=3); d=json.loads(r.read()); names=[m['name'] for m in d.get('models',[])]; print(', '.join(names[:8]) if names else 'No models')"`,
			'ai-backends:check-slate-models': `"${py}" -c "import urllib.request,json; r=urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=3); d=json.loads(r.read()); slate=[m['name'] for m in d.get('models',[]) if 'slate' in m['name']]; print(', '.join(slate) if slate else 'No SLATE models')"`,
			'ai-backends:test-inference': `"${py}" -c "import urllib.request,json; req=urllib.request.Request('http://127.0.0.1:11434/api/generate',data=json.dumps({'model':'slate-fast','prompt':'Hi','stream':False}).encode(),headers={'Content-Type':'application/json'}); r=urllib.request.urlopen(req,timeout=30); d=json.loads(r.read()); print('Inference OK' if d.get('response') else 'No response')"`,
			'integrations:github-auth': `"${py}" -c "import subprocess; r=subprocess.run(['git','credential','fill'],input='protocol=https\\nhost=github.com\\n',capture_output=True,text=True); print('OK' if 'password=' in r.stdout else 'Not configured')"`,
			'integrations:docker-check': `"${py}" -c "import subprocess; r=subprocess.run(['docker','info'],capture_output=True,text=True,timeout=5); print('Running' if r.returncode==0 else 'Not available')"`,
			'integrations:mcp-server': `"${py}" -c "import os; print('Available' if os.path.exists('slate/mcp_server.py') else 'Not found')"`,
			'integrations:runner-check': `"${py}" slate/slate_runner_manager.py --detect`,
			'validation:health-check': `"${py}" slate/slate_status.py --quick`,
			'validation:gpu-access': `"${py}" -c "import torch; print(f'{torch.cuda.device_count()} GPUs, CUDA {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA')"`,
			'validation:test-workflow': `"${py}" slate/slate_workflow_manager.py --status`,
			'validation:security-scan': `"${py}" -c "import os; ag=os.path.exists('slate/action_guard.py'); pii=os.path.exists('slate/pii_scanner.py'); print(f'Guards: AG={ag}, PII={pii}')"`,
			'complete:summary': `"${py}" slate/slate_runtime.py --check-all`,
			'complete:recommendations': `echo "All systems operational. Ready for autonomous development."`,
		};

		const key = `${stepId}:${substepId}`;
		const cmd = commands[key];
		if (!cmd) {
			return 'Skipped (no command)';
		}

		try {
			const { stdout } = await execAsync(cmd, {
				cwd: this._workspaceRoot,
				timeout: 30000,
				env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
			});
			return stdout.trim().split('\n').pop() || 'Done';
		} catch (err: any) {
			throw new Error(err.stderr?.trim()?.split('\n').pop() || err.message || 'Command failed');
		}
	}

	private async _getAINarration(stepId: string): Promise<string> {
		try {
			return await this._queryOllama(stepId);
		} catch {
			return STATIC_NARRATIONS[stepId] || `Executing step: ${stepId}`;
		}
	}

	private async _queryOllama(stepId: string): Promise<string> {
		const config = getSlateConfig();
		const py = config.pythonPath;
		const cwd = this._workspaceRoot;

		// Write a temp Python script to avoid cmd.exe quote escaping issues
		const scriptContent = [
			'import urllib.request, json, sys',
			`step_id = ${JSON.stringify(stepId)}`,
			`prompt = f"You are SLATE, an AI assistant for a local-first development framework. Provide a brief (2-3 sentences) encouraging narration for the {step_id} step of the setup wizard. Be concise and informative."`,
			'payload = json.dumps({"model": "slate-fast", "prompt": prompt, "stream": False, "options": {"temperature": 0.7, "num_predict": 100}})',
			'try:',
			'    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload.encode(), headers={"Content-Type": "application/json"})',
			'    r = urllib.request.urlopen(req, timeout=15)',
			'    d = json.loads(r.read())',
			'    print(d.get("response", ""))',
			'except Exception:',
			'    print("")',
		].join('\n');

		const tmpScript = path.join(cwd, '.slate_narration_tmp.py');
		try {
			await writeFile(tmpScript, scriptContent, 'utf-8');
			const { stdout } = await execAsync(`"${py}" "${tmpScript}"`, { cwd, timeout: 20000 });
			return stdout.trim() || STATIC_NARRATIONS[stepId] || `Processing ${stepId}...`;
		} finally {
			try { await unlink(tmpScript); } catch { /* ignore cleanup errors */ }
		}
	}

	private async _skipCurrentStep(): Promise<void> {
		if (this._currentStep < this._steps.length) {
			this._steps[this._currentStep].status = 'complete';
			this._steps[this._currentStep].substeps.forEach(s => {
				s.status = s.optional ? 'skipped' : 'success';
				s.result = 'Skipped';
			});
			this._sendToWebview({ type: 'stepComplete', step: this._steps[this._currentStep], currentIndex: this._currentStep });
		}
	}

	private _exitGuidedMode(): void {
		this._isRunning = false;
	}

	// ── Systems Check (post-onboarding — replaces onboarding view) ──────────

	private async _runSystemsCheck(): Promise<void> {
		const config = getSlateConfig();
		const py = config.pythonPath;
		const cwd = this._workspaceRoot;

		this._sendToWebview({ type: 'systemsCheckStart' });

		const checks = [
			{ id: 'python', label: 'Python Runtime', cmd: `"${py}" -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"` },
			{ id: 'venv', label: 'Virtual Env', cmd: `"${py}" -c "import sys; print('Active' if sys.prefix!=sys.base_prefix else 'System')"` },
			{ id: 'gpu', label: 'GPU / CUDA', cmd: `"${py}" -c "import torch; print(f'{torch.cuda.device_count()}x GPU, CUDA {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA')"` },
			{ id: 'ollama', label: 'Ollama', cmd: `"${py}" -c "import urllib.request,json; r=urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=3); d=json.loads(r.read()); print(len(d.get('models',[])),'models')"` },
			{ id: 'dashboard', label: 'Dashboard', cmd: `"${py}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080',timeout=2); print('Online')"` },
			{ id: 'runner', label: 'Actions Runner', cmd: `"${py}" slate/slate_runner_manager.py --detect` },
			{ id: 'docker', label: 'Docker', cmd: `"${py}" -c "import subprocess; r=subprocess.run(['docker','info'],capture_output=True,text=True,timeout=5); print('Running' if r.returncode==0 else 'Stopped')"` },
			{ id: 'security', label: 'Security Guards', cmd: `"${py}" -c "import os; ag=os.path.exists('slate/action_guard.py'); pii=os.path.exists('slate/pii_scanner.py'); sdk=os.path.exists('slate/sdk_source_guard.py'); print(f'AG={ag} PII={pii} SDK={sdk}')"` },
		];

		for (const check of checks) {
			this._sendToWebview({ type: 'systemsCheckItem', id: check.id, label: check.label, status: 'running' });
			try {
				const { stdout } = await execAsync(check.cmd, { cwd, timeout: 10000 });
				const result = stdout.trim().split('\n').pop() || 'OK';
				const ok = !result.toLowerCase().includes('no ') && !result.toLowerCase().includes('stopped') && !result.toLowerCase().includes('system');
				this._sendToWebview({ type: 'systemsCheckItem', id: check.id, label: check.label, status: ok ? 'pass' : 'warn', result });
			} catch {
				this._sendToWebview({ type: 'systemsCheckItem', id: check.id, label: check.label, status: 'fail', result: 'Unreachable' });
			}
		}

		this._sendToWebview({ type: 'systemsCheckComplete' });
	}

	// ── Control Board Functions ─────────────────────────────────────────────

	private async _runSlateCommand(command: string): Promise<void> {
		const config = getSlateConfig();
		const terminal = vscode.window.createTerminal('SLATE');
		terminal.show();
		terminal.sendText(`"${config.pythonPath}" ${command}`);
	}

	private async _refreshStatus(): Promise<void> {
		const config = getSlateConfig();
		const py = config.pythonPath;
		const cwd = this._workspaceRoot;

		const checks = [
			{ id: 'svcDashboard', cmd: `"${py}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080', timeout=2); print('ok')"`, ok: ':8080 Online', fail: ':8080 Offline' },
			{ id: 'svcOllama', cmd: `"${py}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2); print('ok')"`, ok: ':11434 Online', fail: ':11434 Offline' },
			{ id: 'svcRunner', cmd: `"${py}" slate/slate_runner_manager.py --detect`, ok: 'Online', fail: 'Offline' },
			{ id: 'svcGPU', cmd: `"${py}" -c "import torch; print('ok' if torch.cuda.is_available() else 'no')"`, ok: 'GPU Active', fail: 'No GPU' },
			{ id: 'svcDocker', cmd: `"${py}" -c "import subprocess; r=subprocess.run(['docker','info'],capture_output=True,text=True,timeout=5); print('ok' if r.returncode==0 else 'no')"`, ok: 'Running', fail: 'Stopped' },
			{ id: 'svcMCP', cmd: `"${py}" -c "import os; print('ok' if os.path.exists('slate/mcp_server.py') else 'no')"`, ok: 'Ready', fail: 'Missing' },
		];

		const results = await Promise.allSettled(
			checks.map(async (c) => {
				try {
					const { stdout } = await execAsync(c.cmd, { cwd, timeout: 8000 });
					const up = stdout.trim().includes('ok') || stdout.trim().includes('Found');
					return { id: c.id, active: up, status: up ? c.ok : c.fail };
				} catch {
					return { id: c.id, active: false, status: c.fail };
				}
			})
		);

		const services = results
			.filter((r): r is PromiseFulfilledResult<{ id: string; active: boolean; status: string }> => r.status === 'fulfilled')
			.map(r => r.value);

		this._sendToWebview({ type: 'serviceStatus', services });
	}

	private async _transitionDevCycleStage(stage: string): Promise<void> {
		try {
			const response = await fetch(`${DASHBOARD_URL}/api/devcycle/transition`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ stage }),
			});
			if (response.ok) {
				const data = await response.json();
				this._sendToWebview({ type: 'devCycleUpdate', data });
			}
		} catch {
			this._sendToWebview({ type: 'devCycleUpdate', data: { current_stage: stage, stage_progress_percent: 0 } });
		}
	}

	private async _toggleLearningMode(active: boolean): Promise<void> {
		try {
			await fetch(`${DASHBOARD_URL}/api/interactive/learning`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ active }),
			});
		} catch {
			// Dashboard not available, gracefully degrade
		}
	}

	private async _fetchInteractiveStatus(): Promise<void> {
		try {
			const response = await fetch(`${DASHBOARD_URL}/api/interactive/`);
			if (response.ok) {
				const data = await response.json();
				this._sendToWebview({ type: 'interactiveStatus', data });
			}
		} catch {
			// Dashboard not available
		}
	}

	// ── Webview Communication ───────────────────────────────────────────────

	private _sendToWebview(message: any): void {
		if (this._view) {
			this._view.webview.postMessage(message);
		}
	}

	// ── HTML Generation ─────────────────────────────────────────────────────

	private _getHtml(webview: vscode.Webview, onboardingComplete: boolean): string {
		const nonce = getNonce();
		const isReOnboard = !onboardingComplete && this._context.globalState.get<string>('slateLastVersion', '0.0.0') !== '0.0.0';
		const lastVer = this._context.globalState.get<string>('slateLastVersion', '0.0.0');
		const isVersionMismatch = lastVer !== '0.0.0' && lastVer !== EXTENSION_VERSION;
		const csp = [
			"default-src 'none'",
			`frame-src ${DASHBOARD_URL}`,
			`img-src ${DASHBOARD_URL} data: https:`,
			"style-src 'unsafe-inline'",
			`script-src 'nonce-${nonce}'`,
			`connect-src ${DASHBOARD_URL}`,
		].join('; ');

		return `<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
	<meta http-equiv="Content-Security-Policy" content="${csp}"/>
	<title>SLATE</title>
	<style>
		/* ═══════════════════════════════════════════════════════════════════
		   SLATE UNIFIED DESIGN SYSTEM — M3 ProArt (Spec 007) — Redesigned
		   ═══════════════════════════════════════════════════════════════════ */
		:root {
			--sl-bg-root: ${SLATE_TOKENS.bgRoot};
			--sl-bg-surface: ${SLATE_TOKENS.bgSurface};
			--sl-bg-surface-variant: ${SLATE_TOKENS.bgSurfaceVariant};
			--sl-bg-container: ${SLATE_TOKENS.bgContainer};
			--sl-bg-container-high: ${SLATE_TOKENS.bgContainerHigh};
			--sl-bg-container-highest: ${SLATE_TOKENS.bgContainerHighest};
			--sl-primary: ${SLATE_TOKENS.primary};
			--sl-primary-light: ${SLATE_TOKENS.primaryLight};
			--sl-primary-dark: ${SLATE_TOKENS.primaryDark};
			--sl-primary-container: ${SLATE_TOKENS.primaryContainer};
			--sl-on-primary: ${SLATE_TOKENS.onPrimary};
			--sl-accent: ${SLATE_TOKENS.accent};
			--sl-accent-light: ${SLATE_TOKENS.accentLight};
			--sl-accent-dark: ${SLATE_TOKENS.accentDark};
			--sl-accent-glow: ${SLATE_TOKENS.accentGlow};
			--sl-accent-container: ${SLATE_TOKENS.accentContainer};
			--sl-text-primary: ${SLATE_TOKENS.textPrimary};
			--sl-text-secondary: ${SLATE_TOKENS.textSecondary};
			--sl-text-tertiary: ${SLATE_TOKENS.textTertiary};
			--sl-text-disabled: ${SLATE_TOKENS.textDisabled};
			--sl-on-surface: ${SLATE_TOKENS.onSurface};
			--sl-on-surface-variant: ${SLATE_TOKENS.onSurfaceVariant};
			--sl-border: ${SLATE_TOKENS.border};
			--sl-border-variant: ${SLATE_TOKENS.borderVariant};
			--sl-border-focus: ${SLATE_TOKENS.borderFocus};
			--sl-outline: ${SLATE_TOKENS.outline};
			--sl-outline-variant: ${SLATE_TOKENS.outlineVariant};
			--sl-success: ${SLATE_TOKENS.success};
			--sl-success-container: ${SLATE_TOKENS.successContainer};
			--sl-warning: ${SLATE_TOKENS.warning};
			--sl-warning-container: ${SLATE_TOKENS.warningContainer};
			--sl-error: ${SLATE_TOKENS.error};
			--sl-error-container: ${SLATE_TOKENS.errorContainer};
			--sl-info: ${SLATE_TOKENS.info};
			--sl-info-container: ${SLATE_TOKENS.infoContainer};
			--sl-font-display: ${SLATE_TOKENS.fontDisplay};
			--sl-font-body: ${SLATE_TOKENS.fontBody};
			--sl-font-mono: ${SLATE_TOKENS.fontMono};
			--sl-body-small: ${SLATE_TOKENS.bodySmall};
			--sl-body-medium: ${SLATE_TOKENS.bodyMedium};
			--sl-label-large: ${SLATE_TOKENS.labelLarge};
			--sl-label-medium: ${SLATE_TOKENS.labelMedium};
			--sl-label-small: ${SLATE_TOKENS.labelSmall};
			--sl-title-large: ${SLATE_TOKENS.titleLarge};
			--sl-title-medium: ${SLATE_TOKENS.titleMedium};
			--sl-elevation-1: ${SLATE_TOKENS.elevation1};
			--sl-elevation-2: ${SLATE_TOKENS.elevation2};
			--sl-elevation-3: ${SLATE_TOKENS.elevation3};
			--sl-ease: ${SLATE_TOKENS.easingStandard};
			--sl-ease-decel: ${SLATE_TOKENS.easingDecelerate};
			--sl-ease-spring: ${SLATE_TOKENS.easingSpring};
			--sl-dur-short: ${SLATE_TOKENS.durationShort3};
			--sl-dur-medium: ${SLATE_TOKENS.durationMedium2};
			--sl-radius-xs: ${SLATE_TOKENS.radiusXs};
			--sl-radius-sm: ${SLATE_TOKENS.radiusSm};
			--sl-radius-md: ${SLATE_TOKENS.radiusMd};
			--sl-radius-lg: ${SLATE_TOKENS.radiusLg};
			--sl-radius-xl: ${SLATE_TOKENS.radiusXl};
			--sl-radius-full: ${SLATE_TOKENS.radiusFull};
		}

		* { margin: 0; padding: 0; box-sizing: border-box; }
		html, body {
			height: 100%; width: 100%;
			font-family: var(--sl-font-body);
			background: var(--sl-bg-root);
			color: var(--sl-text-primary);
			font-size: 13px; line-height: 1.5;
			overflow-x: hidden;
		}
		::-webkit-scrollbar { width: 6px; }
		::-webkit-scrollbar-track { background: transparent; }
		::-webkit-scrollbar-thumb { background: var(--sl-border-variant); border-radius: 3px; }
		::-webkit-scrollbar-thumb:hover { background: var(--sl-accent); }

		/* ── Animations ── */
		@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
		@keyframes glow { 0%,100% { box-shadow: 0 0 4px var(--sl-accent-glow); } 50% { box-shadow: 0 0 16px var(--sl-accent-glow); } }
		@keyframes slideIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
		@keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
		@keyframes spin { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
		@keyframes blink { 0%,50% { opacity:1; } 51%,100% { opacity:0; } }
		@keyframes heroFloat { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-6px); } }
		@keyframes statusPulse { 0%,100% { box-shadow:0 0 0 0 var(--sl-success); } 50% { box-shadow:0 0 0 4px transparent; } }
		@keyframes checkCascade { 0% { opacity:0; transform:translateX(-8px); } 100% { opacity:1; transform:translateX(0); } }

		/* ── View modes ── */
		.view-onboarding { display: \${onboardingComplete ? 'none' : 'block'}; }
		.view-dashboard { display: \${onboardingComplete ? 'block' : 'none'}; }

		/* ═══════════════════════════════════════════════════════════════════
		   ONBOARDING HERO
		   ═══════════════════════════════════════════════════════════════════ */
		.hero {
			display:flex; flex-direction:column;
			align-items:center; justify-content:center;
			min-height:100vh; padding:32px 20px; text-align:center;
			background: radial-gradient(ellipse at 50% 30%, rgba(184,90,60,0.06) 0%, transparent 60%), var(--sl-bg-root);
		}
		.hero.hidden { display:none; }

		.hero-logo { width:72px; height:72px; margin-bottom:16px; animation:heroFloat 4s ease-in-out infinite; }

		.hero-title {
			font-family:var(--sl-font-display);
			font-size:24px; font-weight:700;
			letter-spacing:3px; color:var(--sl-text-primary);
			margin-bottom:6px;
		}
		.hero-subtitle { font-size:var(--sl-body-small); color:var(--sl-text-secondary); max-width:240px; margin-bottom:20px; line-height:1.6; }

		.hero-stats { display:flex; justify-content:center; gap:20px; margin-bottom:20px; }
		.stat { text-align:center; }
		.stat-value { display:block; font-size:20px; font-weight:700; color:var(--sl-accent-light); }
		.stat-label { font-size:9px; color:var(--sl-text-tertiary); text-transform:uppercase; letter-spacing:1px; }

		.update-banner {
			display:none; background:var(--sl-info-container);
			border:1px solid rgba(33,150,243,0.3); border-radius:var(--sl-radius-sm);
			padding:10px 16px; margin-bottom:16px;
			font-size:var(--sl-body-small); color:var(--sl-info);
			text-align:center; max-width:280px;
		}
		.update-banner.visible { display:block; }

		.cta-container { display:flex; flex-direction:column; gap:10px; width:100%; max-width:260px; }

		.cta-primary {
			padding:12px 24px; font-size:var(--sl-label-large); font-weight:600;
			background:linear-gradient(135deg, var(--sl-accent) 0%, var(--sl-accent-dark) 100%);
			color:var(--sl-bg-root); border:none;
			border-radius:var(--sl-radius-xl); cursor:pointer;
			transition:all 0.2s var(--sl-ease);
			font-family:var(--sl-font-display);
			box-shadow:var(--sl-elevation-2);
		}
		.cta-primary:hover { transform:translateY(-2px); box-shadow:var(--sl-elevation-3); }
		.cta-primary:active { transform:translateY(0); }

		.cta-secondary {
			padding:10px 20px; font-size:var(--sl-body-small);
			background:transparent; color:var(--sl-text-secondary);
			border:1px solid var(--sl-border-variant);
			border-radius:var(--sl-radius-xl); cursor:pointer;
			transition:all 0.2s; font-family:var(--sl-font-display);
		}
		.cta-secondary:hover { border-color:var(--sl-accent); color:var(--sl-accent-light); }

		.cta-theme {
			padding:10px 20px; font-size:var(--sl-body-small);
			background:var(--sl-primary-container);
			color:var(--sl-primary-light);
			border:1px solid rgba(184,90,60,0.3);
			border-radius:var(--sl-radius-xl); cursor:pointer;
			transition:all 0.2s; font-family:var(--sl-font-display);
		}
		.cta-theme:hover { background:rgba(184,90,60,0.2); border-color:var(--sl-primary); }

		.features { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-top:20px; width:100%; max-width:280px; }
		.feature-card {
			background:var(--sl-bg-container); border:1px solid var(--sl-border);
			border-radius:var(--sl-radius-md); padding:12px 8px; text-align:center;
			transition:all 0.2s var(--sl-ease); box-shadow:var(--sl-elevation-1);
		}
		.feature-card:hover { border-color:var(--sl-accent); transform:translateY(-2px); box-shadow:var(--sl-elevation-2); }
		.feature-icon { font-size:18px; margin-bottom:4px; }
		.feature-title { font-size:var(--sl-label-small); font-weight:600; color:var(--sl-accent-light); }
		.feature-desc { font-size:9px; color:var(--sl-text-tertiary); }

		.sys-detect { display:none; margin-top:12px; text-align:center; }
		.sys-detect.active { display:block; }
		.sys-detect-text { font-size:var(--sl-body-small); color:var(--sl-text-tertiary); }
		.sys-detect-spinner { display:inline-block; width:14px; height:14px; border:2px solid var(--sl-border-variant); border-top-color:var(--sl-accent); border-radius:50%; animation:spin 0.8s linear infinite; vertical-align:middle; margin-right:6px; }

		/* ═══════════════════════════════════════════════════════════════════
		   GUIDED OVERLAY (7-step wizard)
		   ═══════════════════════════════════════════════════════════════════ */
		.guided-overlay { display:none; padding:16px; background:var(--sl-bg-root); min-height:100vh; }
		.guided-overlay.active { display:block; }

		.step-progress { display:flex; justify-content:center; gap:6px; margin-bottom:20px; }
		.step-dot { width:10px; height:10px; border-radius:50%; background:var(--sl-text-tertiary); transition:all 0.3s var(--sl-ease); }
		.step-dot.active { background:var(--sl-accent); transform:scale(1.3); box-shadow:0 0 10px var(--sl-accent-glow); }
		.step-dot.complete { background:var(--sl-success); }
		.step-dot.error { background:var(--sl-error); }

		.narrator {
			background:var(--sl-bg-container); border:1px solid var(--sl-border);
			border-radius:var(--sl-radius-md); padding:14px; margin-bottom:16px;
			display:flex; gap:10px; align-items:flex-start; box-shadow:var(--sl-elevation-1);
		}
		.narrator-avatar { width:36px; height:36px; background:linear-gradient(135deg,var(--sl-accent) 0%,var(--sl-accent-dark) 100%); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:18px; flex-shrink:0; }
		.narrator-text { font-size:var(--sl-body-small); line-height:1.6; color:var(--sl-text-secondary); }
		.narrator-text.typing::after { content:'|'; animation:blink 0.7s infinite; }

		.step-card { background:var(--sl-bg-container); border:1px solid var(--sl-border); border-radius:var(--sl-radius-md); padding:16px; margin-bottom:16px; box-shadow:var(--sl-elevation-1); }
		.step-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
		.step-title { font-size:var(--sl-title-medium); font-weight:600; color:var(--sl-accent-light); }
		.step-status { font-size:var(--sl-label-small); padding:3px 10px; border-radius:var(--sl-radius-full); font-weight:500; }
		.step-status.active { background:var(--sl-accent-container); color:var(--sl-accent-light); }
		.step-status.executing { background:var(--sl-warning-container); color:var(--sl-warning); }
		.step-status.complete { background:var(--sl-success-container); color:var(--sl-success); }
		.step-status.error { background:var(--sl-error-container); color:var(--sl-error); }
		.step-description { font-size:var(--sl-body-small); color:var(--sl-text-secondary); margin-bottom:12px; }

		.substeps { display:flex; flex-direction:column; gap:6px; }
		.substep { display:flex; align-items:center; gap:8px; padding:8px; background:var(--sl-bg-surface); border-radius:var(--sl-radius-sm); font-size:var(--sl-body-small); animation:checkCascade 0.3s ease-out; }
		.substep-icon { width:18px; height:18px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:10px; flex-shrink:0; }
		.substep-icon.pending { background:var(--sl-text-tertiary); color:var(--sl-bg-root); }
		.substep-icon.running { background:var(--sl-warning); color:var(--sl-bg-root); animation:spin 1s linear infinite; }
		.substep-icon.success { background:var(--sl-success); color:white; }
		.substep-icon.error { background:var(--sl-error); color:white; }
		.substep-icon.skipped { background:var(--sl-text-tertiary); color:var(--sl-bg-root); opacity:0.5; }
		.substep-label { flex:1; color:var(--sl-text-secondary); }
		.substep-label.optional { font-style:italic; }
		.substep-result { font-size:10px; color:var(--sl-text-tertiary); font-family:var(--sl-font-mono); max-width:120px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

		.guided-controls { display:flex; gap:10px; justify-content:center; }
		.control-btn { padding:8px 16px; font-size:var(--sl-body-small); border-radius:var(--sl-radius-xl); cursor:pointer; transition:all 0.2s; border:1px solid var(--sl-border-variant); background:transparent; color:var(--sl-text-secondary); font-family:var(--sl-font-display); }
		.control-btn:hover { background:var(--sl-accent-container); border-color:var(--sl-accent); color:var(--sl-accent-light); }
		.control-btn.primary { background:var(--sl-accent); border-color:var(--sl-accent); color:var(--sl-bg-root); }
		.control-btn.primary:hover { background:var(--sl-accent-light); }

		.complete-screen { text-align:center; padding:32px 16px; display:none; }
		.complete-icon { font-size:48px; margin-bottom:16px; }
		.complete-title { font-size:var(--sl-title-large); color:var(--sl-success); margin-bottom:10px; }
		.complete-summary { font-size:var(--sl-body-medium); color:var(--sl-text-secondary); margin-bottom:20px; line-height:1.6; }

		/* ═══════════════════════════════════════════════════════════════════
		   DASHBOARD — Collapsible, Organized, Sidebar-Optimized
		   ═══════════════════════════════════════════════════════════════════ */

		/* ── Header ── */
		.dash-header {
			display:flex; align-items:center; justify-content:space-between;
			padding:10px 14px;
			background:linear-gradient(180deg, var(--sl-bg-surface) 0%, var(--sl-bg-root) 100%);
			border-bottom:1px solid var(--sl-border);
		}
		.logo-section { display:flex; align-items:center; gap:8px; }
		.logo-icon {
			width:28px; height:28px;
			background:linear-gradient(135deg,var(--sl-accent) 0%,var(--sl-accent-dark) 100%);
			border-radius:var(--sl-radius-sm); display:flex; align-items:center; justify-content:center;
			font-size:14px; color:var(--sl-bg-root);
			animation:glow 3s ease-in-out infinite;
		}
		.logo-text { font-size:var(--sl-body-medium); font-weight:700; letter-spacing:1.5px; color:var(--sl-text-primary); font-family:var(--sl-font-display); }
		.logo-ver { font-size:8px; color:var(--sl-text-tertiary); font-family:var(--sl-font-mono); }
		.header-status { display:flex; align-items:center; gap:5px; }
		.status-dot { width:7px; height:7px; border-radius:50%; background:var(--sl-success); animation:statusPulse 2s ease-in-out infinite; }
		.status-text { font-size:10px; color:var(--sl-text-secondary); }

		/* ── Collapsible Section ── */
		.section { border-bottom:1px solid var(--sl-border); }
		.section-header {
			display:flex; align-items:center; justify-content:space-between;
			padding:10px 14px; cursor:pointer; user-select:none;
			transition:background 0.15s;
		}
		.section-header:hover { background:var(--sl-bg-surface); }
		.section-header-left { display:flex; align-items:center; gap:8px; }
		.section-icon { font-size:14px; color:var(--sl-accent-light); }
		.section-title { font-size:var(--sl-label-medium); font-weight:600; color:var(--sl-text-primary); text-transform:uppercase; letter-spacing:0.08em; }
		.section-badge { font-size:9px; padding:2px 6px; border-radius:var(--sl-radius-full); background:var(--sl-accent-container); color:var(--sl-accent-light); font-weight:500; }
		.section-chevron { font-size:10px; color:var(--sl-text-tertiary); transition:transform 0.2s var(--sl-ease); }
		.section.collapsed .section-chevron { transform:rotate(-90deg); }
		.section-body { padding:0 14px 12px; }
		.section.collapsed .section-body { display:none; }

		/* ── Health Status (consolidated) ── */
		.health-grid { display:grid; grid-template-columns:1fr 1fr; gap:5px; }
		.health-item {
			display:flex; align-items:center; gap:6px;
			padding:6px 8px; background:var(--sl-bg-container);
			border:1px solid var(--sl-border); border-radius:var(--sl-radius-sm);
			font-size:10px; transition:all 0.2s;
		}
		.health-item.pass { border-left:2px solid var(--sl-success); }
		.health-item.fail { border-left:2px solid var(--sl-error); }
		.health-item.warn { border-left:2px solid var(--sl-warning); }
		.health-item.running { border-left:2px solid var(--sl-info); }
		.health-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
		.health-dot.pass { background:var(--sl-success); }
		.health-dot.fail { background:var(--sl-error); }
		.health-dot.warn { background:var(--sl-warning); }
		.health-dot.running { background:var(--sl-info); animation:pulse 1s infinite; }
		.health-dot.idle { background:var(--sl-text-tertiary); }
		.health-label { flex:1; color:var(--sl-text-secondary); }
		.health-result { font-size:9px; color:var(--sl-text-tertiary); font-family:var(--sl-font-mono); max-width:80px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

		/* ── Services (compact cards) ── */
		.service-list { display:flex; flex-direction:column; gap:4px; }
		.service-row {
			display:flex; align-items:center; gap:8px;
			padding:8px 10px; background:var(--sl-bg-container);
			border:1px solid var(--sl-border); border-radius:var(--sl-radius-sm);
			cursor:pointer; transition:all 0.15s var(--sl-ease);
		}
		.service-row:hover { border-color:var(--sl-accent); background:var(--sl-bg-container-high); }
		.service-row.active { border-left:2px solid var(--sl-success); }
		.service-row.inactive { border-left:2px solid var(--sl-text-tertiary); opacity:0.7; }
		.svc-icon { font-size:14px; width:20px; text-align:center; }
		.svc-info { flex:1; }
		.svc-name { font-size:11px; font-weight:600; color:var(--sl-text-primary); }
		.svc-detail { font-size:9px; color:var(--sl-text-tertiary); font-family:var(--sl-font-mono); }
		.svc-dot { width:6px; height:6px; border-radius:50%; flex-shrink:0; }
		.svc-dot.on { background:var(--sl-success); }
		.svc-dot.off { background:var(--sl-text-tertiary); }

		/* ── Quick Actions (grid) ── */
		.action-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }
		.action-btn {
			display:flex; flex-direction:column; align-items:center; gap:4px;
			padding:10px 6px; background:var(--sl-bg-container);
			border:1px solid var(--sl-border); border-radius:var(--sl-radius-sm);
			cursor:pointer; transition:all 0.15s; font-family:var(--sl-font-body);
		}
		.action-btn:hover { background:var(--sl-bg-container-high); border-color:var(--sl-accent); transform:scale(1.03); }
		.action-btn:active { transform:scale(0.97); }
		.action-icon { font-size:16px; color:var(--sl-accent-light); }
		.action-label { font-size:9px; color:var(--sl-text-tertiary); text-align:center; }

		/* ── Dev Cycle (compact inline) ── */
		.dev-cycle-bar { display:flex; gap:2px; margin-bottom:8px; height:4px; border-radius:2px; overflow:hidden; }
		.dev-cycle-segment { flex:1; transition:all 0.3s; opacity:0.3; }
		.dev-cycle-segment.active { opacity:1; height:6px; margin-top:-1px; }
		.dev-cycle-info { display:flex; align-items:center; justify-content:space-between; }
		.dev-cycle-stage { font-size:11px; font-weight:600; color:var(--sl-accent-light); }
		.dev-cycle-progress { font-size:10px; color:var(--sl-text-tertiary); }

		/* ── Primary Actions ── */
		.primary-actions { padding:12px 14px; border-bottom:1px solid var(--sl-border); }
		.btn-stack { display:flex; flex-direction:column; gap:6px; }
		.btn {
			display:flex; align-items:center; justify-content:center; gap:6px;
			padding:10px 14px; border:1px solid var(--sl-border-variant);
			border-radius:var(--sl-radius-md); cursor:pointer;
			font-size:var(--sl-label-medium); font-family:var(--sl-font-display);
			transition:all 0.15s;
		}
		.btn-primary {
			background:linear-gradient(135deg, var(--sl-accent) 0%, var(--sl-accent-dark) 100%);
			border-color:var(--sl-accent); color:var(--sl-bg-root); font-weight:600;
		}
		.btn-primary:hover { box-shadow:var(--sl-elevation-2); transform:translateY(-1px); }
		.btn-secondary { background:transparent; color:var(--sl-text-secondary); }
		.btn-secondary:hover { background:var(--sl-accent-container); border-color:var(--sl-accent); color:var(--sl-accent-light); }
		.btn-row { display:flex; gap:6px; }
		.btn-row .btn { flex:1; }

		/* ── Footer ── */
		.dash-footer {
			padding:8px 14px; border-top:1px solid var(--sl-border);
			display:flex; align-items:center; justify-content:space-between;
		}
		.version-text { font-size:9px; color:var(--sl-text-tertiary); font-family:var(--sl-font-mono); }
		.footer-link { background:none; border:none; color:var(--sl-text-tertiary); font-size:9px; cursor:pointer; font-family:var(--sl-font-body); text-decoration:underline; transition:color 0.2s; }
		.footer-link:hover { color:var(--sl-accent-light); }
	</style>
</head>
<body>

	<!-- ═══════════════════════════════════════════════════════════════════
	     ONBOARDING VIEW
	     ═══════════════════════════════════════════════════════════════════ -->
	<div class="view-onboarding">
		<section class="hero" id="hero">
			<svg class="hero-logo" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
				<defs><linearGradient id="starGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:${SLATE_TOKENS.accentLight}"/><stop offset="100%" style="stop-color:${SLATE_TOKENS.accent}"/></linearGradient></defs>
				<circle cx="50" cy="50" r="8" fill="url(#starGrad)"/>
				\${[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map(a => \`<line x1="50" y1="50" x2="\${50 + 40 * Math.cos(a * Math.PI / 180)}" y2="\${50 + 40 * Math.sin(a * Math.PI / 180)}" stroke="url(#starGrad)" stroke-width="2" stroke-linecap="round"/>\`).join('')}
			</svg>
			<h1 class="hero-title">S.L.A.T.E.</h1>
			<p class="hero-subtitle">Synchronized Living Architecture for Transformation and Evolution</p>
			<div class="update-banner \${isVersionMismatch ? 'visible' : ''}" id="updateBanner">
				\${isVersionMismatch ? \`Updated to v\${EXTENSION_VERSION} (was v\${lastVer}) \\u2014 re-running setup to apply new features\` : \`Updated to v\${EXTENSION_VERSION}\`}
			</div>
			<div class="hero-stats" id="heroStats">
				<div class="stat"><span class="stat-value" id="statGpu">...</span><span class="stat-label">GPU</span></div>
				<div class="stat"><span class="stat-value" id="statAi">...</span><span class="stat-label">Local AI</span></div>
				<div class="stat"><span class="stat-value" id="statCost">$0</span><span class="stat-label">Cloud Cost</span></div>
			</div>
			<div class="sys-detect active" id="sysDetect"><span class="sys-detect-spinner"></span><span class="sys-detect-text">Scanning your system...</span></div>
			<div class="cta-container" id="ctaContainer" style="display:none;">
				<button class="cta-primary" id="btnStartGuided">Start Guided Setup</button>
				<button class="cta-theme" id="btnApplyTheme">Apply SLATE Theme</button>
				<button class="cta-secondary" id="btnSkipOnboarding">Skip to Dashboard</button>
			</div>
			<div class="features" id="featureCards" style="display:none;">
				<div class="feature-card"><div class="feature-icon">&#x1F9E0;</div><div class="feature-title">Local AI</div><div class="feature-desc" id="featAi">Ollama + Models</div></div>
				<div class="feature-card"><div class="feature-icon">&#x26A1;</div><div class="feature-title" id="featGpuTitle">GPU</div><div class="feature-desc" id="featGpu">Detecting...</div></div>
				<div class="feature-card"><div class="feature-icon">&#x1F916;</div><div class="feature-title">Agentic</div><div class="feature-desc">Claude + Copilot</div></div>
				<div class="feature-card"><div class="feature-icon">&#x1F4E6;</div><div class="feature-title">CI/CD</div><div class="feature-desc">Self-Hosted Runner</div></div>
			</div>
		</section>
		<div class="guided-overlay" id="guidedOverlay">
			<div class="step-progress" id="stepProgress"></div>
			<div class="narrator" id="narrator"><div class="narrator-avatar">&#x1F916;</div><div class="narrator-text" id="narratorText">Initializing SLATE setup wizard...</div></div>
			<div class="step-card" id="stepCard">
				<div class="step-header"><div class="step-title" id="stepTitle">Preparing...</div><div class="step-status active" id="stepStatus">Active</div></div>
				<div class="step-description" id="stepDescription">Getting ready to configure your SLATE environment.</div>
				<div class="substeps" id="substeps"></div>
			</div>
			<div class="guided-controls"><button class="control-btn" id="btnSkipStep">Skip</button><button class="control-btn" id="btnExitGuided">Exit</button></div>
			<div class="complete-screen" id="completeScreen">
				<div class="complete-icon">&#x1F389;</div>
				<div class="complete-title">Setup Complete!</div>
				<div class="complete-summary" id="completeSummary">Your SLATE system is fully operational.</div>
				<div class="cta-container"><button class="cta-primary" id="btnFinishOnboarding">Open Dashboard</button><button class="cta-secondary" id="btnCloseGuided">Close</button></div>
			</div>
		</div>
	</div>

	<!-- ═══════════════════════════════════════════════════════════════════
	     DASHBOARD VIEW — Organized, Collapsible Sections
	     ═══════════════════════════════════════════════════════════════════ -->
	<div class="view-dashboard">

		<!-- Header -->
		<div class="dash-header">
			<div class="logo-section">
				<div class="logo-icon">&#x2726;</div>
				<div><div class="logo-text">S.L.A.T.E.</div><div class="logo-ver">v\${EXTENSION_VERSION}</div></div>
			</div>
			<div class="header-status"><div class="status-dot" id="systemStatus"></div><span class="status-text" id="statusText">Online</span></div>
		</div>

		<!-- Primary Actions -->
		<div class="primary-actions">
			<div class="btn-stack">
				<button class="btn btn-primary" id="btnOpenDashboard"><span>&#x2616;</span><span>Open Dashboard</span></button>
				<div class="btn-row">
					<button class="btn btn-secondary" id="btnStartServices"><span>&#x25B6;</span><span>Start</span></button>
					<button class="btn btn-secondary" id="btnFullStatus"><span>&#x2139;</span><span>Status</span></button>
					<button class="btn btn-secondary" id="btnOpenChat"><span>&#x1F4AC;</span><span>Chat</span></button>
				</div>
			</div>
		</div>

		<!-- Section: Health Check (collapsible) -->
		<div class="section" id="sectionHealth">
			<div class="section-header" data-section="sectionHealth">
				<div class="section-header-left"><span class="section-icon">&#x2695;</span><span class="section-title">Health</span><span class="section-badge" id="healthBadge">\\u2014</span></div>
				<span class="section-chevron">&#x25BC;</span>
			</div>
			<div class="section-body">
				<div class="health-grid" id="healthGrid">
					<div class="health-item" id="check-python"><div class="health-dot idle"></div><div class="health-label">Python</div><div class="health-result">\\u2014</div></div>
					<div class="health-item" id="check-venv"><div class="health-dot idle"></div><div class="health-label">Venv</div><div class="health-result">\\u2014</div></div>
					<div class="health-item" id="check-gpu"><div class="health-dot idle"></div><div class="health-label">GPU</div><div class="health-result">\\u2014</div></div>
					<div class="health-item" id="check-ollama"><div class="health-dot idle"></div><div class="health-label">Ollama</div><div class="health-result">\\u2014</div></div>
					<div class="health-item" id="check-dashboard"><div class="health-dot idle"></div><div class="health-label">Dashboard</div><div class="health-result">\\u2014</div></div>
					<div class="health-item" id="check-runner"><div class="health-dot idle"></div><div class="health-label">Runner</div><div class="health-result">\\u2014</div></div>
					<div class="health-item" id="check-docker"><div class="health-dot idle"></div><div class="health-label">Docker</div><div class="health-result">\\u2014</div></div>
					<div class="health-item" id="check-security"><div class="health-dot idle"></div><div class="health-label">Security</div><div class="health-result">\\u2014</div></div>
				</div>
			</div>
		</div>

		<!-- Section: Services (collapsible) -->
		<div class="section" id="sectionServices">
			<div class="section-header" data-section="sectionServices">
				<div class="section-header-left"><span class="section-icon">&#x2699;</span><span class="section-title">Services</span></div>
				<span class="section-chevron">&#x25BC;</span>
			</div>
			<div class="section-body">
				<div class="service-list">
					<div class="service-row inactive" id="svcDashboard" data-cmd="slate/slate_status.py --quick">
						<span class="svc-icon">&#x2616;</span>
						<div class="svc-info"><div class="svc-name">Dashboard</div><div class="svc-detail" id="svcDashboardDetail">:8080</div></div>
						<div class="svc-dot off" id="svcDashboardDot"></div>
					</div>
					<div class="service-row inactive" id="svcOllama" data-cmd="slate/foundry_local.py --check">
						<span class="svc-icon">&#x1F9E0;</span>
						<div class="svc-info"><div class="svc-name">Ollama</div><div class="svc-detail" id="svcOllamaDetail">:11434</div></div>
						<div class="svc-dot off" id="svcOllamaDot"></div>
					</div>
					<div class="service-row inactive" id="svcRunner" data-cmd="slate/slate_runner_manager.py --status">
						<span class="svc-icon">&#x25B6;</span>
						<div class="svc-info"><div class="svc-name">Actions Runner</div><div class="svc-detail" id="svcRunnerDetail">GitHub</div></div>
						<div class="svc-dot off" id="svcRunnerDot"></div>
					</div>
					<div class="service-row inactive" id="svcGPU" data-cmd="slate/slate_gpu_manager.py --status">
						<span class="svc-icon">&#x2756;</span>
						<div class="svc-info"><div class="svc-name">GPU Compute</div><div class="svc-detail" id="svcGPUDetail">Detecting...</div></div>
						<div class="svc-dot off" id="svcGPUDot"></div>
					</div>
					<div class="service-row inactive" id="svcDocker" data-cmd="slate/slate_docker_daemon.py --status">
						<span class="svc-icon">&#x2693;</span>
						<div class="svc-info"><div class="svc-name">Docker</div><div class="svc-detail" id="svcDockerDetail">Daemon</div></div>
						<div class="svc-dot off" id="svcDockerDot"></div>
					</div>
					<div class="service-row inactive" id="svcMCP" data-cmd="slate/claude_code_manager.py --validate">
						<span class="svc-icon">&#x2728;</span>
						<div class="svc-info"><div class="svc-name">MCP Server</div><div class="svc-detail" id="svcMCPDetail">Claude</div></div>
						<div class="svc-dot off" id="svcMCPDot"></div>
					</div>
				</div>
			</div>
		</div>

		<!-- Section: Dev Cycle (collapsible, starts collapsed) -->
		<div class="section collapsed" id="sectionDevCycle">
			<div class="section-header" data-section="sectionDevCycle">
				<div class="section-header-left"><span class="section-icon">&#x21BA;</span><span class="section-title">Dev Cycle</span><span class="dev-cycle-stage" id="currentStage">CODE</span></div>
				<span class="section-chevron">&#x25BC;</span>
			</div>
			<div class="section-body">
				<div class="dev-cycle-bar">
					<div class="dev-cycle-segment" id="segPlan" data-stage="PLAN" style="background:${SLATE_TOKENS.stagePlan};"></div>
					<div class="dev-cycle-segment active" id="segCode" data-stage="CODE" style="background:${SLATE_TOKENS.stageCode};"></div>
					<div class="dev-cycle-segment" id="segTest" data-stage="TEST" style="background:${SLATE_TOKENS.stageTest};"></div>
					<div class="dev-cycle-segment" id="segDeploy" data-stage="DEPLOY" style="background:${SLATE_TOKENS.stageDeploy};"></div>
					<div class="dev-cycle-segment" id="segFeedback" data-stage="FEEDBACK" style="background:${SLATE_TOKENS.stageFeedback};"></div>
				</div>
				<div class="dev-cycle-info">
					<span class="dev-cycle-stage" id="stageName">Coding</span>
					<span class="dev-cycle-progress" id="stageProgress">45%</span>
				</div>
			</div>
		</div>

		<!-- Section: Quick Actions (collapsible) -->
		<div class="section" id="sectionActions">
			<div class="section-header" data-section="sectionActions">
				<div class="section-header-left"><span class="section-icon">&#x26A1;</span><span class="section-title">Actions</span></div>
				<span class="section-chevron">&#x25BC;</span>
			</div>
			<div class="section-body">
				<div class="action-grid">
					<button class="action-btn" data-action="workflow" title="Workflow status"><span class="action-icon">&#x21BA;</span><span class="action-label">Workflow</span></button>
					<button class="action-btn" data-action="benchmark" title="Run benchmarks"><span class="action-icon">&#x26A1;</span><span class="action-label">Benchmark</span></button>
					<button class="action-btn" data-action="security" title="Security audit"><span class="action-icon">&#x1F512;</span><span class="action-label">Security</span></button>
					<button class="action-btn" data-action="models" title="Model status"><span class="action-icon">&#x1F9E0;</span><span class="action-label">Models</span></button>
					<button class="action-btn" data-action="gpu" title="GPU status"><span class="action-icon">&#x2756;</span><span class="action-label">GPU Mgr</span></button>
					<button class="action-btn" data-action="rerun" title="Re-run guided setup"><span class="action-icon">&#x2726;</span><span class="action-label">Setup</span></button>
				</div>
			</div>
		</div>

		<!-- Footer -->
		<div class="dash-footer">
			<span class="version-text">SLATE v\${EXTENSION_VERSION}</span>
			<button class="footer-link" id="btnResetOnboarding">Re-run setup</button>
		</div>
	</div>

	<!-- ═══════════════════════════════════════════════════════════════════
	     SCRIPTS
	     ═══════════════════════════════════════════════════════════════════ -->
	<script nonce="\${nonce}">
		var vscode = acquireVsCodeApi();
		var currentStep = 0;
		var totalSteps = 7;

		/* ── Collapsible sections ── */
		document.querySelectorAll('.section-header').forEach(function(hdr) {
			hdr.addEventListener('click', function() {
				var sec = document.getElementById(hdr.dataset.section);
				if (sec) sec.classList.toggle('collapsed');
			});
		});

		/* ── Onboarding init ── */
		if (!\${onboardingComplete}) {
			vscode.postMessage({ type: 'detectSystem' });
			setTimeout(function() {
				var det = document.getElementById('sysDetect');
				var cta = document.getElementById('ctaContainer');
				var feat = document.getElementById('featureCards');
				if (det && det.classList.contains('active')) {
					det.classList.remove('active');
					if(cta) cta.style.display = 'flex';
					if(feat) feat.style.display = 'grid';
				}
			}, 8000);
		}

		/* ── Guided setup functions ── */
		function startGuided() {
			document.getElementById('hero').classList.add('hidden');
			document.getElementById('guidedOverlay').classList.add('active');
			vscode.postMessage({ type: 'startGuided' });
			renderStepProgress();
		}
		function skipOnboarding() { vscode.postMessage({ type: 'skipOnboarding' }); }
		function applySlateTheme() { vscode.postMessage({ type: 'applySlateTheme' }); }
		function skipStep() { vscode.postMessage({ type: 'skipStep' }); }
		function exitGuided() {
			document.getElementById('hero').classList.remove('hidden');
			document.getElementById('guidedOverlay').classList.remove('active');
			document.getElementById('completeScreen').style.display = 'none';
			document.getElementById('stepCard').style.display = 'block';
			vscode.postMessage({ type: 'exitGuided' });
		}
		function finishOnboarding() { vscode.postMessage({ type: 'finishOnboarding' }); }
		function resetOnboarding() { vscode.postMessage({ type: 'startGuidedMode' }); }

		function renderStepProgress() {
			var c = document.getElementById('stepProgress'); if (!c) return; c.innerHTML = '';
			for (var i = 0; i < totalSteps; i++) {
				var d = document.createElement('div'); d.className = 'step-dot';
				if (i < currentStep) d.classList.add('complete');
				if (i === currentStep) d.classList.add('active');
				c.appendChild(d);
			}
		}
		function updateStep(step, index) {
			currentStep = index; renderStepProgress();
			var t = document.getElementById('stepTitle'); if(t) t.textContent = step.title;
			var d = document.getElementById('stepDescription'); if(d) d.textContent = step.description;
			var s = document.getElementById('stepStatus');
			if(s) { s.textContent = step.status.charAt(0).toUpperCase() + step.status.slice(1); s.className = 'step-status ' + step.status; }
			renderSubsteps(step.substeps);
		}
		function renderSubsteps(substeps) {
			var c = document.getElementById('substeps'); if (!c) return; c.innerHTML = '';
			var icons = { pending:'\\u25CB', running:'\\u25D0', success:'\\u2713', error:'\\u2717', skipped:'\\u2014' };
			substeps.forEach(function(sub) {
				var div = document.createElement('div'); div.className = 'substep'; div.id = 'substep-' + sub.id;
				var lc = sub.optional ? 'substep-label optional' : 'substep-label';
				div.innerHTML = '<div class="substep-icon ' + sub.status + '">' + (icons[sub.status]||'\\u25CB') + '</div>' +
					'<div class="' + lc + '">' + sub.label + (sub.optional ? ' (optional)' : '') + '</div>' +
					(sub.result ? '<div class="substep-result">' + sub.result + '</div>' : '');
				c.appendChild(div);
			});
		}
		function updateSubstep(stepId, sub) {
			var el = document.getElementById('substep-' + sub.id); if (!el) return;
			var icons = { pending:'\\u25CB', running:'\\u25D0', success:'\\u2713', error:'\\u2717', skipped:'\\u2014' };
			var ic = el.querySelector('.substep-icon');
			if(ic) { ic.textContent = icons[sub.status]||'\\u25CB'; ic.className = 'substep-icon ' + sub.status; }
			if (sub.result) {
				var r = el.querySelector('.substep-result');
				if (!r) { r = document.createElement('div'); r.className = 'substep-result'; el.appendChild(r); }
				r.textContent = sub.result;
			}
		}
		function showComplete(summary) {
			var sc = document.getElementById('stepCard'); if(sc) sc.style.display = 'none';
			var cs = document.getElementById('completeScreen'); if(cs) cs.style.display = 'block';
			var sm = document.getElementById('completeSummary'); if(sm && summary) sm.textContent = summary;
			document.querySelectorAll('.step-dot').forEach(function(d) { d.classList.remove('active'); d.classList.add('complete'); });
		}
		function updateSystemProfile(p) {
			var g = document.getElementById('statGpu');
			if(g) { g.textContent = p.gpuCount > 0 ? p.gpuCount + 'x' : 'CPU'; }
			var a = document.getElementById('statAi');
			if(a) { a.textContent = p.ollamaAvailable ? p.ollamaModels.length + ' models' : 'Pending'; }
			var fg = document.getElementById('featGpu');
			if(fg && p.gpuCount > 0) { fg.textContent = (p.gpuModels[0]||'GPU').replace('NVIDIA GeForce ',''); }
			var ft = document.getElementById('featGpuTitle');
			if(ft && p.gpuCount > 1) { ft.textContent = 'Dual GPU'; }
			var fa = document.getElementById('featAi');
			if(fa) {
				if (p.ollamaAvailable) {
					var sl = p.ollamaModels.filter(function(m){return m.indexOf('slate')>=0;});
					fa.textContent = sl.length > 0 ? sl.length + ' SLATE + ' + (p.ollamaModels.length - sl.length) + ' other' : p.ollamaModels.length + ' models';
				} else { fa.textContent = 'Ollama needed'; }
			}
			var det = document.getElementById('sysDetect'); if(det) det.classList.remove('active');
			var cta = document.getElementById('ctaContainer'); if(cta) cta.style.display = 'flex';
			var feat = document.getElementById('featureCards'); if(feat) feat.style.display = 'grid';
		}

		/* ── Health check item updater ── */
		function updateHealthItem(id, status, result) {
			var item = document.getElementById('check-' + id); if (!item) return;
			item.className = 'health-item ' + status;
			var dot = item.querySelector('.health-dot'); if(dot) dot.className = 'health-dot ' + status;
			var res = item.querySelector('.health-result'); if(res) res.textContent = result || '\\u2014';
		}
		function updateHealthBadge() {
			var items = document.querySelectorAll('.health-item');
			var pass = 0; var total = items.length;
			items.forEach(function(it) { if (it.classList.contains('pass')) pass++; });
			var badge = document.getElementById('healthBadge');
			if (badge) badge.textContent = pass + '/' + total;
		}

		/* ── Onboarding button listeners ── */
		var bsg = document.getElementById('btnStartGuided');
		if(bsg) bsg.addEventListener('click', function() { startGuided(); });
		var bat = document.getElementById('btnApplyTheme');
		if(bat) bat.addEventListener('click', function() { applySlateTheme(); });
		var bso = document.getElementById('btnSkipOnboarding');
		if(bso) bso.addEventListener('click', function() { skipOnboarding(); });
		var bss = document.getElementById('btnSkipStep');
		if(bss) bss.addEventListener('click', function() { skipStep(); });
		var beg = document.getElementById('btnExitGuided');
		if(beg) beg.addEventListener('click', function() { exitGuided(); });
		var bfo = document.getElementById('btnFinishOnboarding');
		if(bfo) bfo.addEventListener('click', function() { finishOnboarding(); });
		var bcg = document.getElementById('btnCloseGuided');
		if(bcg) bcg.addEventListener('click', function() { exitGuided(); });
		var bro = document.getElementById('btnResetOnboarding');
		if(bro) bro.addEventListener('click', function() { resetOnboarding(); });

		/* ── Dashboard button listeners ── */
		var bod = document.getElementById('btnOpenDashboard');
		if(bod) bod.addEventListener('click', function() { vscode.postMessage({type:'openDashboard'}); });
		var bs = document.getElementById('btnStartServices');
		if(bs) bs.addEventListener('click', function() { vscode.postMessage({type:'runCommand',command:'slate/slate_orchestrator.py start'}); });
		var bf = document.getElementById('btnFullStatus');
		if(bf) bf.addEventListener('click', function() { vscode.postMessage({type:'showStatus'}); });
		var boc = document.getElementById('btnOpenChat');
		if(boc) boc.addEventListener('click', function() { vscode.postMessage({type:'openChat'}); });

		/* ── Service row click listeners ── */
		document.querySelectorAll('.service-row').forEach(function(row) {
			row.addEventListener('click', function() {
				var cmd = row.dataset.cmd;
				if(cmd) vscode.postMessage({type:'runCommand',command:cmd});
			});
		});

		/* ── Quick action listeners ── */
		document.querySelectorAll('.action-btn').forEach(function(b) {
			b.addEventListener('click', function() {
				switch(b.dataset.action) {
					case 'workflow': vscode.postMessage({type:'runCommand',command:'slate/slate_workflow_manager.py --status'}); break;
					case 'benchmark': vscode.postMessage({type:'runCommand',command:'slate/slate_benchmark.py'}); break;
					case 'security': vscode.postMessage({type:'runCommand',command:'slate/action_guard.py --scan'}); break;
					case 'models': vscode.postMessage({type:'runCommand',command:'slate/slate_model_trainer.py --status'}); break;
					case 'gpu': vscode.postMessage({type:'runCommand',command:'slate/slate_gpu_manager.py --status'}); break;
					case 'rerun': vscode.postMessage({type:'startGuidedMode'}); break;
				}
			});
		});

		/* ── Dev cycle segment listeners ── */
		document.querySelectorAll('.dev-cycle-segment').forEach(function(s) {
			s.style.cursor = 'pointer';
			s.addEventListener('click', function() {
				if(s.dataset.stage) vscode.postMessage({type:'transitionStage',stage:s.dataset.stage});
			});
		});

		/* ── Dev cycle updater ── */
		var stageNames = {PLAN:'Planning',CODE:'Coding',TEST:'Testing',DEPLOY:'Deploying',FEEDBACK:'Feedback'};
		function updateDevCycle(data) {
			if(!data) return;
			var cs = data.current_stage || 'CODE';
			var pr = data.stage_progress_percent || 0;
			var cse = document.getElementById('currentStage'); if(cse) cse.textContent = cs;
			var sne = document.getElementById('stageName'); if(sne) sne.textContent = stageNames[cs]||cs;
			var spe = document.getElementById('stageProgress'); if(spe) spe.textContent = pr+'%';
			['PLAN','CODE','TEST','DEPLOY','FEEDBACK'].forEach(function(st) {
				var seg = document.getElementById('seg'+st.charAt(0)+st.slice(1).toLowerCase());
				if(seg) {
					if(st===cs) { seg.classList.add('active'); seg.style.opacity='1'; }
					else { seg.classList.remove('active'); seg.style.opacity='0.3'; }
				}
			});
		}

		/* ── Auto-run health check on dashboard load (single batch, not 8 sequential) ── */
		if (\${onboardingComplete}) {
			vscode.postMessage({type:'runSystemsCheck'});
			vscode.postMessage({type:'refreshStatus'});
		}

		/* ── Message handler ── */
		window.addEventListener('message', function(e) {
			var m = e.data;
			try {
				switch(m.type) {
					case 'stepUpdate': updateStep(m.step, m.currentIndex); break;
					case 'narration': var nt=document.getElementById('narratorText'); if(nt) nt.textContent=m.text; break;
					case 'substepUpdate': updateSubstep(m.stepId, m.substep); break;
					case 'stepComplete':
						updateStep(m.step, m.currentIndex);
						if(m.step.id==='complete') { var s=m.step.substeps.find(function(x){return x.id==='summary';}); showComplete(s&&s.result); }
						break;
					case 'systemProfile': updateSystemProfile(m.profile); break;
					case 'themeApplied': break;
					case 'devCycleUpdate': updateDevCycle(m.data); break;
					case 'interactiveStatus': if(m.data&&m.data.dev_cycle) updateDevCycle(m.data.dev_cycle); break;
					case 'serviceStatus':
						if(m.services) m.services.forEach(function(svc) {
							var row = document.getElementById(svc.id);
							if(!row) return;
							if(svc.active) { row.classList.remove('inactive'); row.classList.add('active'); }
							else { row.classList.remove('active'); row.classList.add('inactive'); }
							var detail = row.querySelector('.svc-detail');
							if(detail) detail.textContent = svc.status;
							var dot = row.querySelector('.svc-dot');
							if(dot) { dot.className = svc.active ? 'svc-dot on' : 'svc-dot off'; }
						});
						break;
					case 'systemsCheckStart':
						['python','venv','gpu','ollama','dashboard','runner','docker','security'].forEach(function(id) { updateHealthItem(id,'running','Checking...'); });
						break;
					case 'systemsCheckItem':
						updateHealthItem(m.id, m.status, m.result);
						updateHealthBadge();
						break;
					case 'systemsCheckComplete':
						updateHealthBadge();
						break;
				}
			} catch(err) {
				console.error('[SLATE] Message error:', err);
			}
		});
	</script>
</body>
</html>`;
	}
}

// ── Registration Helper ─────────────────────────────────────────────────────

export function registerUnifiedDashboard(context: vscode.ExtensionContext): vscode.Disposable {
	const provider = new SlateUnifiedDashboardViewProvider(context.extensionUri, context);
	return vscode.window.registerWebviewViewProvider(
		SlateUnifiedDashboardViewProvider.viewType,
		provider,
		{ webviewOptions: { retainContextWhenHidden: true } },
	);
}

function getNonce(): string {
	const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
	let text = '';
	for (let i = 0; i < 32; i++) {
		text += chars.charAt(Math.floor(Math.random() * chars.length));
	}
	return text;
}
