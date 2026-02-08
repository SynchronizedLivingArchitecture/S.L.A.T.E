// Modified: 2026-02-07T18:00:00Z | Author: COPILOT | Change: Unified dashboard combining guided setup, control board, and dashboard into one view
/**
 * SLATE Unified Dashboard View
 * ==============================
 * Single integrated webview that combines:
 * - Guided onboarding (shown first for new users)
 * - Control board (service status, dev cycle, learning mode)
 * - Dashboard iframe (FastAPI backend at 127.0.0.1:8080)
 *
 * Design System: SLATE M3 ProArt (black/copper/warm-white)
 * - Primary: #0a0a0a (true black substrate)
 * - Accent: #B87333 (copper/bronze)
 * - Text: #F5F0EB (warm white)
 * - Semantic: sage green, warm amber, muted rose, steel blue
 */

import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';
import { getSlateConfig } from './extension';

const execAsync = promisify(exec);

const DASHBOARD_URL = 'http://127.0.0.1:8080';

// ── Types ───────────────────────────────────────────────────────────────────

interface GuidedStep {
	id: string;
	title: string;
	description: string;
	status: 'pending' | 'active' | 'executing' | 'complete' | 'error';
	substeps: SubStep[];
	aiNarration?: string;
	duration?: number;
}

interface SubStep {
	id: string;
	label: string;
	status: 'pending' | 'running' | 'success' | 'error';
	result?: string;
}

// ── SLATE Design Tokens (M3 ProArt) ────────────────────────────────────────

const SLATE_TOKENS = {
	// Surfaces — true black foundation
	bgRoot: '#050505',
	bgSurface: '#0a0a0a',
	bgContainer: '#111111',
	bgContainerHigh: '#1a1a1a',
	bgContainerHighest: '#222222',

	// Accent — copper/bronze
	accent: '#B87333',
	accentLight: '#C9956B',
	accentDark: '#8B5E2B',
	accentGlow: 'rgba(184,115,51,0.15)',
	accentContainer: 'rgba(184,115,51,0.12)',

	// Text — warm white
	textPrimary: '#F5F0EB',
	textSecondary: '#A8A29E',
	textTertiary: '#78716C',
	textDisabled: '#44403C',

	// Borders
	border: 'rgba(255,255,255,0.08)',
	borderVariant: 'rgba(255,255,255,0.12)',
	borderFocus: '#B87333',

	// Semantic
	success: '#78B89A',
	warning: '#D4A054',
	error: '#C47070',
	info: '#7EA8BE',

	// Typography
	fontSans: "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
	fontMono: "'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace",

	// Motion
	easeStandard: 'cubic-bezier(0.2, 0, 0, 1)',
	easeDecelerate: 'cubic-bezier(0, 0, 0, 1)',
	durationShort: '150ms',
	durationMedium: '300ms',
	durationLong: '500ms',

	// Radii
	radiusSm: '8px',
	radiusMd: '12px',
	radiusLg: '16px',
	radiusFull: '9999px',

	// Dev cycle stage colors
	stagePlan: '#7EA8BE',
	stageCode: '#B87333',
	stageTest: '#D4A054',
	stageDeploy: '#78B89A',
	stageFeedback: '#9B89B3',
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
			{ id: 'detect-docker', label: 'Detect Docker daemon', status: 'pending' },
			{ id: 'detect-github', label: 'Detect GitHub CLI', status: 'pending' },
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
			{ id: 'start-dashboard', label: 'Start dashboard server', status: 'pending' },
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
			{ id: 'check-models', label: 'Check installed models', status: 'pending' },
			{ id: 'pull-mistral', label: 'Ensure mistral-nemo available', status: 'pending' },
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
			{ id: 'docker-check', label: 'Check Docker daemon', status: 'pending' },
			{ id: 'mcp-server', label: 'Validate MCP server', status: 'pending' },
			{ id: 'runner-check', label: 'Check GitHub Actions runner', status: 'pending' },
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

// ── Static AI narrations (fallback) ─────────────────────────────────────────

const STATIC_NARRATIONS: Record<string, string> = {
	'welcome': "Welcome to S.L.A.T.E. — your Synchronized Living Architecture for Transformation and Evolution. I'll guide you through the setup process, scanning your system and configuring everything for optimal performance.",
	'system-scan': "Now scanning your system for prerequisites. I'm looking for Python, GPU drivers, Ollama, Docker, and GitHub integration. This helps me tailor the setup to your hardware.",
	'core-services': "Setting up the core SLATE infrastructure. This includes the Python virtual environment, essential dependencies, the dashboard server, and the service orchestrator.",
	'ai-backends': "Configuring your local AI inference stack. SLATE runs entirely on your hardware — no cloud APIs, no data leaving your machine. I'll verify Ollama is connected and models are ready.",
	'integrations': "Connecting to external services — GitHub for CI/CD, Docker for containerization, and the MCP server for AI agent bridging. All connections use 127.0.0.1 only.",
	'validation': "Running comprehensive validation across all subsystems. I'll check service health, GPU access, workflow dispatch capability, and security configuration.",
	'complete': "Excellent! Your SLATE system is fully operational. You now have a local-first AI development environment with dual-GPU inference, autonomous task execution, and full CI/CD integration.",
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

		// Check onboarding state
		const onboardingComplete = this._context.globalState.get<boolean>('slateOnboardingComplete', false);
		webviewView.webview.html = this._getHtml(webviewView.webview, onboardingComplete);

		// Handle messages
		webviewView.webview.onDidReceiveMessage(async (message) => {
			switch (message.type) {
				// ── Guided onboarding messages ──
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

				// ── Control board messages ──
				case 'runCommand':
					await this._runSlateCommand(message.command);
					break;
				case 'refreshStatus':
					await this._refreshStatus();
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

				// ── Dashboard messages ──
				case 'openExternal':
					void vscode.env.openExternal(vscode.Uri.parse(DASHBOARD_URL));
					break;
				case 'openPanel':
					await vscode.commands.executeCommand('slate.openDashboard');
					break;
			}
		});

		// Auto-refresh status every 30 seconds (only in dashboard mode)
		this._statusInterval = setInterval(() => {
			if (this._context.globalState.get<boolean>('slateOnboardingComplete', false)) {
				this._refreshStatus();
				this._fetchInteractiveStatus();
			}
		}, 30000);

		webviewView.onDidDispose(() => {
			if (this._statusInterval) {
				clearInterval(this._statusInterval);
			}
		});

		// Initial status fetch if onboarding complete
		if (onboardingComplete) {
			this._refreshStatus();
			this._fetchInteractiveStatus();
		}
	}

	// ── Onboarding management ───────────────────────────────────────────────

	private async _completeOnboarding(): Promise<void> {
		await this._context.globalState.update('slateOnboardingComplete', true);
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
		if (this._view) {
			this._view.webview.html = this._getHtml(this._view.webview, false);
		}
	}

	public refresh(): void {
		if (this._view) {
			const onboardingComplete = this._context.globalState.get<boolean>('slateOnboardingComplete', false);
			this._view.webview.html = this._getHtml(this._view.webview, onboardingComplete);
		}
	}

	// ── Guided Installation Flow ────────────────────────────────────────────

	private async _startGuidedInstall(): Promise<void> {
		if (this._isRunning) { return; }
		this._isRunning = true;
		this._currentStep = 0;
		this._steps = JSON.parse(JSON.stringify(GUIDED_STEPS));

		for (let i = 0; i < this._steps.length; i++) {
			this._currentStep = i;
			await this._executeStep(this._steps[i]);
			if (!this._isRunning) { break; }
		}

		this._isRunning = false;
	}

	private async _executeStep(step: GuidedStep): Promise<void> {
		step.status = 'active';

		// Send narration
		const narration = await this._getAINarration(step.id);
		this._sendToWebview({ type: 'narration', text: narration });
		this._sendToWebview({ type: 'stepUpdate', step, currentIndex: this._currentStep });

		// Execute all substeps
		step.status = 'executing';
		this._sendToWebview({ type: 'stepUpdate', step, currentIndex: this._currentStep });

		for (const substep of step.substeps) {
			await this._executeSubstep(step, substep);
			if (!this._isRunning) { return; }
		}

		step.status = 'complete';
		this._sendToWebview({ type: 'stepComplete', step, currentIndex: this._currentStep });

		// Auto-advance after a brief pause
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
			substep.status = 'error';
			substep.result = err?.message || 'Failed';
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
			'system-scan:detect-gpu': `"${py}" -c "import subprocess; r=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True); print(r.stdout.strip() or 'No GPU')"`,
			'system-scan:detect-ollama': `"${py}" -c "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=3); print('Connected')"`,
			'system-scan:detect-docker': `"${py}" -c "import subprocess; r=subprocess.run(['docker','info'],capture_output=True,text=True); print('Running' if r.returncode==0 else 'Not running')"`,
			'system-scan:detect-github': `"${py}" -c "import subprocess; r=subprocess.run(['git','credential','fill'],input='protocol=https\\nhost=github.com\\n',capture_output=True,text=True); print('Authenticated' if 'password=' in r.stdout else 'Not configured')"`,
			'core-services:init-venv': `"${py}" -c "import sys,os; print('venv active' if sys.prefix!=sys.base_prefix else 'system python')"`,
			'core-services:install-deps': `"${py}" -c "import pkg_resources; print(f'{len(list(pkg_resources.working_set))} packages installed')"`,
			'core-services:start-dashboard': `"${py}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080',timeout=3); print('Dashboard running')"`,
			'core-services:init-orchestrator': `"${py}" slate/slate_orchestrator.py status`,
			'ai-backends:check-ollama': `"${py}" -c "import urllib.request,json; r=urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=3); d=json.loads(r.read()); print(f'{len(d.get(\"models\",[]))} models available')"`,
			'ai-backends:check-models': `"${py}" -c "import urllib.request,json; r=urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=3); d=json.loads(r.read()); names=[m['name'] for m in d.get('models',[])]; print(', '.join(names[:5]))"`,
			'ai-backends:pull-mistral': `"${py}" -c "import urllib.request,json; r=urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=3); d=json.loads(r.read()); names=[m['name'] for m in d.get('models',[])]; print('Available' if any('mistral' in n for n in names) else 'Need to pull')"`,
			'ai-backends:test-inference': `"${py}" -c "import urllib.request,json; req=urllib.request.Request('http://127.0.0.1:11434/api/generate',data=json.dumps({'model':'slate-fast','prompt':'Hi','stream':False}).encode(),headers={'Content-Type':'application/json'}); r=urllib.request.urlopen(req,timeout=30); d=json.loads(r.read()); print('Inference OK' if d.get('response') else 'No response')"`,
			'integrations:github-auth': `"${py}" -c "import subprocess; r=subprocess.run(['git','credential','fill'],input='protocol=https\\nhost=github.com\\n',capture_output=True,text=True); print('OK' if 'password=' in r.stdout else 'Not configured')"`,
			'integrations:docker-check': `"${py}" -c "import subprocess; r=subprocess.run(['docker','info'],capture_output=True,text=True); print('Running' if r.returncode==0 else 'Not available')"`,
			'integrations:mcp-server': `"${py}" -c "import importlib.util; print('Available' if importlib.util.find_spec('slate.mcp_server') or __import__('os').path.exists('slate/mcp_server.py') else 'Not found')"`,
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
		// Try Ollama first, fall back to static
		try {
			return await this._queryOllama(stepId);
		} catch {
			return STATIC_NARRATIONS[stepId] || `Executing step: ${stepId}`;
		}
	}

	private async _queryOllama(stepId: string): Promise<string> {
		const prompt = `You are SLATE, an AI assistant for a local-first development framework. Provide a brief (2-3 sentences) encouraging narration for the "${stepId}" step of the setup wizard. Be concise and informative.`;

		const payload = JSON.stringify({
			model: 'mistral-nemo',
			prompt,
			stream: false,
			options: { temperature: 0.7, num_predict: 100 },
		});

		const { stdout } = await execAsync(
			`"${getSlateConfig().pythonPath}" -c "import urllib.request,json; req=urllib.request.Request('http://127.0.0.1:11434/api/generate',data=${JSON.stringify(payload)}.encode(),headers={'Content-Type':'application/json'}); r=urllib.request.urlopen(req,timeout=15); d=json.loads(r.read()); print(d.get('response',''))"`,
			{ cwd: this._workspaceRoot, timeout: 20000 },
		);

		return stdout.trim() || STATIC_NARRATIONS[stepId] || `Processing ${stepId}...`;
	}

	private async _skipCurrentStep(): Promise<void> {
		if (this._currentStep < this._steps.length) {
			this._steps[this._currentStep].status = 'complete';
			this._steps[this._currentStep].substeps.forEach(s => {
				s.status = 'success';
				s.result = 'Skipped';
			});
			this._sendToWebview({ type: 'stepComplete', step: this._steps[this._currentStep], currentIndex: this._currentStep });
		}
	}

	private _exitGuidedMode(): void {
		this._isRunning = false;
	}

	// ── Control Board Functions ─────────────────────────────────────────────

	private async _runSlateCommand(command: string): Promise<void> {
		const config = getSlateConfig();
		const terminal = vscode.window.createTerminal('SLATE');
		terminal.show();
		terminal.sendText(`"${config.pythonPath}" ${command}`);
	}

	private async _refreshStatus(): Promise<void> {
		this._sendToWebview({ type: 'statusUpdate' });
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
			// Dashboard not available, just update UI
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
			// Ignore if dashboard unavailable
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
		/* ════════════════════════════════════════════════════════════════
		   SLATE UNIFIED DESIGN SYSTEM — M3 ProArt
		   ════════════════════════════════════════════════════════════════ */
		:root {
			/* Surfaces */
			--sl-bg-root: ${SLATE_TOKENS.bgRoot};
			--sl-bg-surface: ${SLATE_TOKENS.bgSurface};
			--sl-bg-container: ${SLATE_TOKENS.bgContainer};
			--sl-bg-container-high: ${SLATE_TOKENS.bgContainerHigh};
			--sl-bg-container-highest: ${SLATE_TOKENS.bgContainerHighest};

			/* Accent */
			--sl-accent: ${SLATE_TOKENS.accent};
			--sl-accent-light: ${SLATE_TOKENS.accentLight};
			--sl-accent-dark: ${SLATE_TOKENS.accentDark};
			--sl-accent-glow: ${SLATE_TOKENS.accentGlow};
			--sl-accent-container: ${SLATE_TOKENS.accentContainer};

			/* Text */
			--sl-text-primary: ${SLATE_TOKENS.textPrimary};
			--sl-text-secondary: ${SLATE_TOKENS.textSecondary};
			--sl-text-tertiary: ${SLATE_TOKENS.textTertiary};

			/* Borders */
			--sl-border: ${SLATE_TOKENS.border};
			--sl-border-variant: ${SLATE_TOKENS.borderVariant};
			--sl-border-focus: ${SLATE_TOKENS.borderFocus};

			/* Semantic */
			--sl-success: ${SLATE_TOKENS.success};
			--sl-warning: ${SLATE_TOKENS.warning};
			--sl-error: ${SLATE_TOKENS.error};
			--sl-info: ${SLATE_TOKENS.info};

			/* Dev cycle */
			--sl-stage-plan: ${SLATE_TOKENS.stagePlan};
			--sl-stage-code: ${SLATE_TOKENS.stageCode};
			--sl-stage-test: ${SLATE_TOKENS.stageTest};
			--sl-stage-deploy: ${SLATE_TOKENS.stageDeploy};
			--sl-stage-feedback: ${SLATE_TOKENS.stageFeedback};

			/* Typography */
			--sl-font-sans: ${SLATE_TOKENS.fontSans};
			--sl-font-mono: ${SLATE_TOKENS.fontMono};

			/* Motion */
			--sl-ease: ${SLATE_TOKENS.easeStandard};
			--sl-duration-short: ${SLATE_TOKENS.durationShort};
			--sl-duration-medium: ${SLATE_TOKENS.durationMedium};
			--sl-duration-long: ${SLATE_TOKENS.durationLong};

			/* Radii */
			--sl-radius-sm: ${SLATE_TOKENS.radiusSm};
			--sl-radius-md: ${SLATE_TOKENS.radiusMd};
			--sl-radius-lg: ${SLATE_TOKENS.radiusLg};
			--sl-radius-full: ${SLATE_TOKENS.radiusFull};
		}

		* { margin: 0; padding: 0; box-sizing: border-box; }

		html, body {
			height: 100%;
			width: 100%;
			font-family: var(--sl-font-sans);
			background: var(--sl-bg-root);
			color: var(--sl-text-primary);
			font-size: 13px;
			line-height: 1.5;
			overflow-x: hidden;
		}

		/* ── Scrollbar ── */
		::-webkit-scrollbar { width: 6px; }
		::-webkit-scrollbar-track { background: transparent; }
		::-webkit-scrollbar-thumb { background: var(--sl-border-variant); border-radius: 3px; }
		::-webkit-scrollbar-thumb:hover { background: var(--sl-accent); }

		/* ════════════════════════════════════════════════════════════════
		   ANIMATIONS
		   ════════════════════════════════════════════════════════════════ */
		@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
		@keyframes glow { 0%, 100% { box-shadow: 0 0 4px var(--sl-accent-glow); } 50% { box-shadow: 0 0 16px var(--sl-accent-glow); } }
		@keyframes slideIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
		@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
		@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
		@keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }
		@keyframes ringPulse { 0%, 100% { filter: drop-shadow(0 0 4px currentColor); } 50% { filter: drop-shadow(0 0 12px currentColor); } }
		@keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
		@keyframes breathe { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.02); } }
		@keyframes statusPulse { 0%, 100% { box-shadow: 0 0 0 0 var(--sl-success); } 50% { box-shadow: 0 0 0 4px transparent; } }
		@keyframes heroFloat { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }

		/* ════════════════════════════════════════════════════════════════
		   VIEW MODES: Onboarding vs Dashboard
		   ════════════════════════════════════════════════════════════════ */
		.view-onboarding { display: ${onboardingComplete ? 'none' : 'block'}; }
		.view-dashboard { display: ${onboardingComplete ? 'block' : 'none'}; }

		/* ════════════════════════════════════════════════════════════════
		   ONBOARDING HERO
		   ════════════════════════════════════════════════════════════════ */
		.hero {
			display: flex;
			flex-direction: column;
			align-items: center;
			justify-content: center;
			min-height: 100vh;
			padding: 32px 20px;
			text-align: center;
			background:
				radial-gradient(ellipse at 50% 30%, rgba(184,115,51,0.06) 0%, transparent 60%),
				var(--sl-bg-root);
		}

		.hero.hidden { display: none; }

		.hero-logo {
			width: 80px;
			height: 80px;
			margin-bottom: 20px;
			animation: heroFloat 4s ease-in-out infinite;
		}

		.hero-title {
			font-size: 28px;
			font-weight: 700;
			letter-spacing: 3px;
			color: var(--sl-text-primary);
			margin-bottom: 8px;
		}

		.hero-subtitle {
			font-size: 12px;
			color: var(--sl-text-secondary);
			max-width: 240px;
			margin-bottom: 24px;
			line-height: 1.6;
		}

		.hero-stats {
			display: flex;
			justify-content: center;
			gap: 24px;
			margin-bottom: 28px;
		}

		.stat { text-align: center; }
		.stat-value { display: block; font-size: 22px; font-weight: 700; color: var(--sl-accent-light); }
		.stat-label { font-size: 10px; color: var(--sl-text-tertiary); text-transform: uppercase; letter-spacing: 1px; }

		.cta-container {
			display: flex;
			flex-direction: column;
			gap: 10px;
			width: 100%;
			max-width: 260px;
		}

		.cta-primary {
			padding: 12px 24px;
			font-size: 14px;
			font-weight: 600;
			background: linear-gradient(135deg, var(--sl-accent) 0%, var(--sl-accent-dark) 100%);
			color: var(--sl-bg-root);
			border: none;
			border-radius: var(--sl-radius-sm);
			cursor: pointer;
			transition: all 0.2s;
			font-family: var(--sl-font-sans);
			box-shadow: 0 4px 12px rgba(184, 115, 51, 0.25);
		}
		.cta-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(184, 115, 51, 0.35); }
		.cta-primary:active { transform: translateY(0); }

		.cta-secondary {
			padding: 10px 20px;
			font-size: 12px;
			background: transparent;
			color: var(--sl-text-secondary);
			border: 1px solid var(--sl-border-variant);
			border-radius: var(--sl-radius-sm);
			cursor: pointer;
			transition: all 0.2s;
			font-family: var(--sl-font-sans);
		}
		.cta-secondary:hover { border-color: var(--sl-accent); color: var(--sl-accent-light); }

		/* Feature cards in hero */
		.features {
			display: grid;
			grid-template-columns: repeat(2, 1fr);
			gap: 10px;
			margin-top: 24px;
			width: 100%;
			max-width: 300px;
		}
		.feature-card {
			background: var(--sl-bg-container);
			border: 1px solid var(--sl-border);
			border-radius: var(--sl-radius-sm);
			padding: 14px 10px;
			text-align: center;
			transition: all 0.2s;
		}
		.feature-card:hover { border-color: var(--sl-accent); transform: translateY(-2px); }
		.feature-icon { font-size: 20px; margin-bottom: 6px; }
		.feature-title { font-size: 11px; font-weight: 600; color: var(--sl-accent-light); }
		.feature-desc { font-size: 9px; color: var(--sl-text-tertiary); }

		/* ════════════════════════════════════════════════════════════════
		   GUIDED OVERLAY (7-step wizard)
		   ════════════════════════════════════════════════════════════════ */
		.guided-overlay {
			display: none;
			padding: 20px;
			background: var(--sl-bg-root);
			min-height: 100vh;
		}
		.guided-overlay.active { display: block; }

		.step-progress {
			display: flex;
			justify-content: center;
			gap: 6px;
			margin-bottom: 24px;
		}
		.step-dot {
			width: 10px; height: 10px;
			border-radius: 50%;
			background: var(--sl-text-tertiary);
			transition: all 0.3s;
		}
		.step-dot.active { background: var(--sl-info); transform: scale(1.3); box-shadow: 0 0 10px var(--sl-info); }
		.step-dot.complete { background: var(--sl-success); }
		.step-dot.error { background: var(--sl-error); }

		.narrator {
			background: var(--sl-bg-container);
			border: 1px solid var(--sl-border);
			border-radius: var(--sl-radius-md);
			padding: 16px;
			margin-bottom: 20px;
			display: flex;
			gap: 12px;
			align-items: flex-start;
		}
		.narrator-avatar {
			width: 40px; height: 40px;
			background: linear-gradient(135deg, var(--sl-accent) 0%, var(--sl-accent-dark) 100%);
			border-radius: 50%;
			display: flex; align-items: center; justify-content: center;
			font-size: 20px; flex-shrink: 0;
		}
		.narrator-text { font-size: 13px; line-height: 1.6; color: var(--sl-text-secondary); }
		.narrator-text.typing::after { content: '|'; animation: blink 0.7s infinite; }

		.step-card {
			background: var(--sl-bg-container);
			border: 1px solid var(--sl-border);
			border-radius: var(--sl-radius-md);
			padding: 20px;
			margin-bottom: 20px;
		}
		.step-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
		.step-title { font-size: 16px; font-weight: 600; color: var(--sl-accent-light); }
		.step-status { font-size: 11px; padding: 3px 10px; border-radius: var(--sl-radius-full); font-weight: 500; }
		.step-status.active { background: rgba(126,168,190,0.15); color: var(--sl-info); }
		.step-status.executing { background: rgba(212,160,84,0.15); color: var(--sl-warning); }
		.step-status.complete { background: rgba(120,184,154,0.15); color: var(--sl-success); }
		.step-status.error { background: rgba(196,112,112,0.15); color: var(--sl-error); }
		.step-description { font-size: 12px; color: var(--sl-text-secondary); margin-bottom: 16px; }

		.substeps { display: flex; flex-direction: column; gap: 8px; }
		.substep {
			display: flex; align-items: center; gap: 10px;
			padding: 10px;
			background: var(--sl-bg-surface);
			border-radius: var(--sl-radius-sm);
			font-size: 12px;
		}
		.substep-icon {
			width: 18px; height: 18px;
			border-radius: 50%;
			display: flex; align-items: center; justify-content: center;
			font-size: 10px; flex-shrink: 0;
		}
		.substep-icon.pending { background: var(--sl-text-tertiary); color: var(--sl-bg-root); }
		.substep-icon.running { background: var(--sl-warning); color: var(--sl-bg-root); animation: spin 1s linear infinite; }
		.substep-icon.success { background: var(--sl-success); color: white; }
		.substep-icon.error { background: var(--sl-error); color: white; }
		.substep-label { flex: 1; color: var(--sl-text-secondary); }
		.substep-result { font-size: 10px; color: var(--sl-text-tertiary); font-family: var(--sl-font-mono); max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

		.guided-controls { display: flex; gap: 10px; justify-content: center; }
		.control-btn {
			padding: 8px 16px; font-size: 12px;
			border-radius: var(--sl-radius-sm);
			cursor: pointer; transition: all 0.2s;
			border: 1px solid var(--sl-border-variant);
			background: transparent; color: var(--sl-text-secondary);
			font-family: var(--sl-font-sans);
		}
		.control-btn:hover { background: var(--sl-accent-container); border-color: var(--sl-accent); color: var(--sl-accent-light); }
		.control-btn.primary { background: var(--sl-accent); border-color: var(--sl-accent); color: var(--sl-bg-root); }
		.control-btn.primary:hover { background: var(--sl-accent-light); }

		/* Complete screen */
		.complete-screen { text-align: center; padding: 40px 20px; display: none; }
		.complete-icon { font-size: 56px; margin-bottom: 20px; }
		.complete-title { font-size: 22px; color: var(--sl-success); margin-bottom: 12px; }
		.complete-summary { font-size: 13px; color: var(--sl-text-secondary); margin-bottom: 24px; line-height: 1.6; }

		/* ════════════════════════════════════════════════════════════════
		   DASHBOARD MODE — Header
		   ════════════════════════════════════════════════════════════════ */
		.dash-header {
			display: flex;
			align-items: center;
			justify-content: space-between;
			padding: 10px 12px;
			background: var(--sl-bg-surface);
			border-bottom: 1px solid var(--sl-border);
		}
		.logo-section { display: flex; align-items: center; gap: 8px; }
		.logo-icon {
			width: 28px; height: 28px;
			background: linear-gradient(135deg, var(--sl-accent) 0%, var(--sl-accent-dark) 100%);
			border-radius: var(--sl-radius-sm);
			display: flex; align-items: center; justify-content: center;
			font-size: 14px; color: var(--sl-bg-root);
			animation: glow 3s ease-in-out infinite;
		}
		.logo-text { font-size: 13px; font-weight: 700; letter-spacing: 1.5px; color: var(--sl-text-primary); }
		.logo-subtitle { font-size: 8px; color: var(--sl-text-tertiary); text-transform: uppercase; letter-spacing: 0.5px; }

		.status-badge {
			display: flex; align-items: center; gap: 6px;
			padding: 4px 10px;
			background: var(--sl-bg-container);
			border-radius: var(--sl-radius-full);
			font-size: 10px; color: var(--sl-text-secondary);
		}
		.status-dot {
			width: 7px; height: 7px;
			border-radius: 50%;
			background: var(--sl-success);
			animation: statusPulse 2s ease-in-out infinite;
		}

		/* ════════════════════════════════════════════════════════════════
		   DASHBOARD MODE — Service Grid
		   ════════════════════════════════════════════════════════════════ */
		.services {
			display: grid;
			grid-template-columns: repeat(3, 1fr);
			gap: 6px;
			padding: 10px 12px;
		}
		.service-card {
			display: flex; flex-direction: column;
			align-items: center; gap: 4px;
			padding: 10px 6px;
			background: var(--sl-bg-container);
			border: 1px solid var(--sl-border);
			border-radius: var(--sl-radius-sm);
			cursor: pointer;
			transition: all 0.2s var(--sl-ease);
			text-align: center;
			animation: slideIn 0.3s ease-out;
		}
		.service-card:hover { border-color: var(--sl-accent); transform: translateY(-2px); background: var(--sl-bg-container-high); }
		.service-card.active { border-color: rgba(120,184,154,0.3); }
		.service-card.active .service-status { color: var(--sl-success); }
		.service-icon { font-size: 16px; color: var(--sl-accent-light); }
		.service-name { font-size: 10px; font-weight: 600; color: var(--sl-text-primary); }
		.service-status { font-size: 9px; color: var(--sl-text-tertiary); }

		/* ════════════════════════════════════════════════════════════════
		   DASHBOARD MODE — Dev Cycle Ring
		   ════════════════════════════════════════════════════════════════ */
		.dev-cycle-section {
			padding: 10px 12px;
			border-top: 1px solid var(--sl-border);
			background: var(--sl-bg-surface);
		}
		.dev-cycle-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
		.section-title { font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--sl-text-tertiary); }
		.dev-cycle-stage { font-size: 11px; font-weight: 600; color: var(--sl-accent-light); }
		.dev-cycle-ring-container { display: flex; align-items: center; justify-content: center; gap: 14px; }
		.mini-ring { width: 72px; height: 72px; position: relative; }
		.mini-ring svg { width: 100%; height: 100%; }
		.stage-segment { fill: none; stroke-width: 6; stroke-linecap: round; transition: all 0.3s ease; cursor: pointer; }
		.stage-segment.active { stroke-width: 8; filter: drop-shadow(0 0 4px currentColor); animation: ringPulse 2s ease-in-out infinite; }
		.stage-label { fill: var(--sl-text-tertiary); }
		.stage-label.active { fill: var(--sl-accent-light); font-weight: 600; }
		.stage-info { flex: 1; max-width: 110px; }
		.stage-name { font-size: 13px; font-weight: 600; color: var(--sl-text-primary); margin-bottom: 3px; }
		.stage-progress-text { font-size: 10px; color: var(--sl-text-tertiary); margin-bottom: 4px; }
		.stage-bar { height: 3px; background: var(--sl-border); border-radius: 2px; overflow: hidden; }
		.stage-bar-fill { height: 100%; background: linear-gradient(90deg, var(--sl-accent), var(--sl-accent-light)); border-radius: 2px; transition: width 0.5s ease-out; }

		/* ════════════════════════════════════════════════════════════════
		   DASHBOARD MODE — Controls & Quick Actions
		   ════════════════════════════════════════════════════════════════ */
		.controls {
			padding: 10px 12px;
			border-top: 1px solid var(--sl-border);
			display: flex; flex-direction: column; gap: 6px;
		}
		.btn {
			display: flex; align-items: center; justify-content: center; gap: 6px;
			padding: 8px 12px;
			border: 1px solid var(--sl-border-variant);
			border-radius: var(--sl-radius-sm);
			cursor: pointer;
			font-size: 11px;
			font-family: var(--sl-font-sans);
			transition: all 0.2s;
		}
		.btn-primary { background: var(--sl-accent); border-color: var(--sl-accent); color: var(--sl-bg-root); font-weight: 600; }
		.btn-primary:hover { background: var(--sl-accent-light); }
		.btn-secondary { background: transparent; color: var(--sl-text-secondary); }
		.btn-secondary:hover { background: var(--sl-accent-container); border-color: var(--sl-accent); color: var(--sl-accent-light); }
		.btn-row { display: flex; gap: 6px; }
		.btn-row .btn { flex: 1; }

		.quick-actions {
			padding: 10px 12px;
			border-top: 1px solid var(--sl-border);
			background: var(--sl-bg-surface);
		}
		.action-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px; }
		.action-btn {
			display: flex; flex-direction: column; align-items: center; gap: 3px;
			padding: 8px 4px;
			background: var(--sl-bg-container);
			border: 1px solid var(--sl-border);
			border-radius: var(--sl-radius-sm);
			cursor: pointer;
			transition: all 0.2s;
			font-family: var(--sl-font-sans);
		}
		.action-btn:hover { background: var(--sl-bg-container-high); border-color: var(--sl-accent); transform: scale(1.05); }
		.action-btn:active { transform: scale(0.95); }
		.action-icon { font-size: 14px; color: var(--sl-accent-light); }
		.action-label { font-size: 8px; color: var(--sl-text-tertiary); }

		/* ════════════════════════════════════════════════════════════════
		   DASHBOARD MODE — Learning Section
		   ════════════════════════════════════════════════════════════════ */
		.learning-section {
			padding: 8px 12px;
			border-top: 1px solid var(--sl-border);
			display: flex; align-items: center; justify-content: space-between;
			background: linear-gradient(135deg, rgba(184,115,51,0.04) 0%, transparent 100%);
		}
		.learning-info { display: flex; align-items: center; gap: 8px; }
		.learning-icon {
			width: 24px; height: 24px;
			background: var(--sl-accent);
			border-radius: var(--sl-radius-sm);
			display: flex; align-items: center; justify-content: center;
			font-size: 12px;
		}
		.learning-level { font-size: 11px; font-weight: 600; color: var(--sl-text-primary); }
		.learning-xp { font-size: 9px; color: var(--sl-text-tertiary); }
		.learning-toggle {
			position: relative; width: 38px; height: 20px;
			background: var(--sl-border-variant);
			border-radius: 10px;
			cursor: pointer;
			transition: background 0.2s;
		}
		.learning-toggle.active { background: var(--sl-accent); }
		.learning-toggle::after {
			content: '';
			position: absolute; top: 2px; left: 2px;
			width: 16px; height: 16px;
			background: white;
			border-radius: 50%;
			transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
		}
		.learning-toggle.active::after { transform: translateX(18px); }

		/* ════════════════════════════════════════════════════════════════
		   DASHBOARD MODE — Embedded Dashboard iframe
		   ════════════════════════════════════════════════════════════════ */
		.dashboard-frame-section {
			border-top: 1px solid var(--sl-border);
		}
		.dashboard-frame-header {
			display: flex; align-items: center; justify-content: space-between;
			padding: 6px 12px;
			background: var(--sl-bg-surface);
		}
		.frame-actions { display: flex; gap: 4px; }
		.frame-btn {
			background: var(--sl-bg-container);
			color: var(--sl-text-tertiary);
			border: 1px solid var(--sl-border);
			border-radius: var(--sl-radius-sm);
			padding: 3px 8px;
			cursor: pointer;
			font-size: 10px;
			font-family: var(--sl-font-sans);
			transition: all 0.2s;
		}
		.frame-btn:hover { background: var(--sl-bg-container-high); color: var(--sl-accent-light); }

		.dashboard-iframe {
			width: 100%;
			height: 400px;
			border: none;
			background: var(--sl-bg-root);
		}

		.offline-notice {
			display: none;
			padding: 24px;
			text-align: center;
			color: var(--sl-text-tertiary);
			font-size: 12px;
		}
		.offline-notice.visible { display: block; }

		/* ════════════════════════════════════════════════════════════════
		   ONBOARDING RESET BUTTON (in dashboard mode)
		   ════════════════════════════════════════════════════════════════ */
		.reset-onboarding {
			padding: 6px 12px;
			border-top: 1px solid var(--sl-border);
			text-align: center;
		}
		.reset-btn {
			background: none; border: none;
			color: var(--sl-text-tertiary);
			font-size: 9px; cursor: pointer;
			font-family: var(--sl-font-sans);
			text-decoration: underline;
			transition: color 0.2s;
		}
		.reset-btn:hover { color: var(--sl-accent-light); }

		/* ════════════════════════════════════════════════════════════════
		   SCANLINE OVERLAY (subtle)
		   ════════════════════════════════════════════════════════════════ */
		.scanline {
			position: fixed; top: 0; left: 0; right: 0; bottom: 0;
			pointer-events: none;
			background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.02) 2px, rgba(0,0,0,0.02) 4px);
			opacity: 0.4;
			z-index: 9999;
		}
	</style>
</head>
<body>
	<div class="scanline"></div>

	<!-- ══════════════════════════════════════════════════════════════════
	     ONBOARDING VIEW
	     ══════════════════════════════════════════════════════════════════ -->
	<div class="view-onboarding">
		<!-- Hero Section -->
		<section class="hero" id="hero">
			<svg class="hero-logo" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
				<defs>
					<linearGradient id="starGrad" x1="0%" y1="0%" x2="100%" y2="100%">
						<stop offset="0%" style="stop-color:${SLATE_TOKENS.accentLight}"/>
						<stop offset="100%" style="stop-color:${SLATE_TOKENS.accent}"/>
					</linearGradient>
				</defs>
				<circle cx="50" cy="50" r="8" fill="url(#starGrad)"/>
				${[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330].map(angle =>
					`<line x1="50" y1="50" x2="${50 + 40 * Math.cos(angle * Math.PI / 180)}" y2="${50 + 40 * Math.sin(angle * Math.PI / 180)}" stroke="url(#starGrad)" stroke-width="2" stroke-linecap="round"/>`
				).join('')}
			</svg>

			<h1 class="hero-title">S.L.A.T.E.</h1>
			<p class="hero-subtitle">Synchronized Living Architecture for Transformation and Evolution</p>

			<div class="hero-stats">
				<div class="stat"><span class="stat-value">2x</span><span class="stat-label">RTX 5070 Ti</span></div>
				<div class="stat"><span class="stat-value">100%</span><span class="stat-label">Local AI</span></div>
				<div class="stat"><span class="stat-value">$0</span><span class="stat-label">Cloud Cost</span></div>
			</div>

			<div class="cta-container">
				<button class="cta-primary" onclick="startGuided()">Start Guided Setup</button>
				<button class="cta-secondary" onclick="skipOnboarding()">Skip to Dashboard</button>
			</div>

			<div class="features">
				<div class="feature-card"><div class="feature-icon">&#x1F9E0;</div><div class="feature-title">Local AI</div><div class="feature-desc">Ollama + Foundry</div></div>
				<div class="feature-card"><div class="feature-icon">&#x26A1;</div><div class="feature-title">Dual GPU</div><div class="feature-desc">RTX Blackwell</div></div>
				<div class="feature-card"><div class="feature-icon">&#x1F916;</div><div class="feature-title">Agentic</div><div class="feature-desc">Claude + Copilot</div></div>
				<div class="feature-card"><div class="feature-icon">&#x1F4E6;</div><div class="feature-title">CI/CD</div><div class="feature-desc">Self-Hosted Runner</div></div>
			</div>
		</section>

		<!-- Guided Install Overlay -->
		<div class="guided-overlay" id="guidedOverlay">
			<div class="step-progress" id="stepProgress"></div>

			<div class="narrator" id="narrator">
				<div class="narrator-avatar">&#x1F916;</div>
				<div class="narrator-text" id="narratorText">Initializing SLATE setup wizard...</div>
			</div>

			<div class="step-card" id="stepCard">
				<div class="step-header">
					<div class="step-title" id="stepTitle">Preparing...</div>
					<div class="step-status active" id="stepStatus">Active</div>
				</div>
				<div class="step-description" id="stepDescription">Getting ready to configure your SLATE environment.</div>
				<div class="substeps" id="substeps"></div>
			</div>

			<div class="guided-controls">
				<button class="control-btn" onclick="skipStep()">Skip</button>
				<button class="control-btn" onclick="exitGuided()">Exit</button>
			</div>

			<!-- Complete Screen -->
			<div class="complete-screen" id="completeScreen">
				<div class="complete-icon">&#x1F389;</div>
				<div class="complete-title">Setup Complete!</div>
				<div class="complete-summary" id="completeSummary">Your SLATE system is fully operational.</div>
				<div class="cta-container">
					<button class="cta-primary" onclick="finishOnboarding()">Open Dashboard</button>
					<button class="cta-secondary" onclick="exitGuided()">Close</button>
				</div>
			</div>
		</div>
	</div>

	<!-- ══════════════════════════════════════════════════════════════════
	     DASHBOARD VIEW (post-onboarding)
	     ══════════════════════════════════════════════════════════════════ -->
	<div class="view-dashboard">
		<!-- Header -->
		<div class="dash-header">
			<div class="logo-section">
				<div class="logo-icon">&#x2726;</div>
				<div>
					<div class="logo-text">S.L.A.T.E.</div>
					<div class="logo-subtitle">Control Board</div>
				</div>
			</div>
			<div class="status-badge">
				<div class="status-dot" id="systemStatus"></div>
				<span id="statusText">Online</span>
			</div>
		</div>

		<!-- Service Status Grid -->
		<div class="services">
			<div class="service-card active" id="svcDashboard" data-cmd="slate/slate_status.py --quick">
				<div class="service-icon">&#x2616;</div>
				<div class="service-name">Dashboard</div>
				<div class="service-status">:8080</div>
			</div>
			<div class="service-card active" id="svcOllama" data-cmd="slate/foundry_local.py --check">
				<div class="service-icon">&#x2699;</div>
				<div class="service-name">Ollama</div>
				<div class="service-status">:11434</div>
			</div>
			<div class="service-card" id="svcRunner" data-cmd="slate/slate_runner_manager.py --status">
				<div class="service-icon">&#x25B6;</div>
				<div class="service-name">Runner</div>
				<div class="service-status">GitHub</div>
			</div>
			<div class="service-card active" id="svcGPU" data-cmd="slate/slate_gpu_manager.py --status">
				<div class="service-icon">&#x2756;</div>
				<div class="service-name">GPU</div>
				<div class="service-status">2x RTX</div>
			</div>
			<div class="service-card" id="svcDocker" data-cmd="slate/slate_docker_daemon.py --status">
				<div class="service-icon">&#x2693;</div>
				<div class="service-name">Docker</div>
				<div class="service-status">Daemon</div>
			</div>
			<div class="service-card" id="svcMCP" data-cmd="slate/claude_code_manager.py --validate">
				<div class="service-icon">&#x2728;</div>
				<div class="service-name">MCP</div>
				<div class="service-status">Claude</div>
			</div>
		</div>

		<!-- Dev Cycle Ring -->
		<div class="dev-cycle-section">
			<div class="dev-cycle-header">
				<span class="section-title">Development Cycle</span>
				<span class="dev-cycle-stage" id="currentStage">CODE</span>
			</div>
			<div class="dev-cycle-ring-container">
				<div class="mini-ring">
					<svg viewBox="0 0 100 100" id="devCycleRing">
						<circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="6"/>
						<path class="stage-segment" id="segPlan" stroke="${SLATE_TOKENS.stagePlan}" opacity="0.4" d="M 50 10 A 40 40 0 0 1 88.04 30.98" data-stage="PLAN"/>
						<path class="stage-segment active" id="segCode" stroke="${SLATE_TOKENS.stageCode}" d="M 88.04 30.98 A 40 40 0 0 1 80.90 76.18" data-stage="CODE"/>
						<path class="stage-segment" id="segTest" stroke="${SLATE_TOKENS.stageTest}" opacity="0.4" d="M 80.90 76.18 A 40 40 0 0 1 19.10 76.18" data-stage="TEST"/>
						<path class="stage-segment" id="segDeploy" stroke="${SLATE_TOKENS.stageDeploy}" opacity="0.4" d="M 19.10 76.18 A 40 40 0 0 1 11.96 30.98" data-stage="DEPLOY"/>
						<path class="stage-segment" id="segFeedback" stroke="${SLATE_TOKENS.stageFeedback}" opacity="0.4" d="M 11.96 30.98 A 40 40 0 0 1 50 10" data-stage="FEEDBACK"/>
						<text x="50" y="48" text-anchor="middle" class="stage-label active" font-size="10">CODE</text>
						<text x="50" y="60" text-anchor="middle" class="stage-label" font-size="8">45%</text>
					</svg>
				</div>
				<div class="stage-info">
					<div class="stage-name" id="stageName">Coding</div>
					<div class="stage-progress-text" id="stageProgress">45% complete</div>
					<div class="stage-bar"><div class="stage-bar-fill" id="stageBarFill" style="width:45%"></div></div>
				</div>
			</div>
		</div>

		<!-- Learning Mode -->
		<div class="learning-section">
			<div class="learning-info">
				<div class="learning-icon">&#x1F393;</div>
				<div>
					<div class="learning-level" id="learningLevel">Level 2</div>
					<div class="learning-xp" id="learningXP">175 XP</div>
				</div>
			</div>
			<div class="learning-toggle" id="learningToggle" title="Toggle Learning Mode"></div>
		</div>

		<!-- Main Controls -->
		<div class="controls">
			<button class="btn btn-primary" id="btnStartServices">
				<span>&#x25B6;</span><span>Start Services</span>
			</button>
			<div class="btn-row">
				<button class="btn btn-secondary" id="btnFullStatus">
					<span>&#x2139;</span><span>Full Status</span>
				</button>
				<button class="btn btn-secondary" id="btnRerunSetup">
					<span>&#x2726;</span><span>Re-run Setup</span>
				</button>
			</div>
		</div>

		<!-- Quick Actions -->
		<div class="quick-actions">
			<div class="section-title" style="margin-bottom:6px;">Quick Actions</div>
			<div class="action-grid">
				<button class="action-btn" data-action="chat" title="Open @slate chat">
					<span class="action-icon">&#x1F4AC;</span><span class="action-label">Chat</span>
				</button>
				<button class="action-btn" data-action="workflow" title="Workflow status">
					<span class="action-icon">&#x21BA;</span><span class="action-label">Workflow</span>
				</button>
				<button class="action-btn" data-action="benchmark" title="Run benchmarks">
					<span class="action-icon">&#x26A1;</span><span class="action-label">Bench</span>
				</button>
				<button class="action-btn" data-action="security" title="Security audit">
					<span class="action-icon">&#x1F512;</span><span class="action-label">Security</span>
				</button>
			</div>
		</div>

		<!-- Embedded Dashboard -->
		<div class="dashboard-frame-section">
			<div class="dashboard-frame-header">
				<span class="section-title">Dashboard</span>
				<div class="frame-actions">
					<button class="frame-btn" id="btnRefreshFrame" title="Refresh">&#x21BB;</button>
					<button class="frame-btn" id="btnExpandFrame" title="Open in panel">&#x2197;</button>
					<button class="frame-btn" id="btnExternalFrame" title="Open in browser">&#x2756;</button>
				</div>
			</div>
			<iframe class="dashboard-iframe" id="dashboardFrame" src="${DASHBOARD_URL}" title="SLATE Dashboard"></iframe>
			<div class="offline-notice" id="offlineNotice">
				<p>Dashboard server not reachable at ${DASHBOARD_URL}</p>
				<button class="cta-secondary" onclick="retryDashboard()" style="margin-top:10px;">Retry Connection</button>
			</div>
		</div>

		<!-- Reset Onboarding -->
		<div class="reset-onboarding">
			<button class="reset-btn" onclick="resetOnboarding()">Re-run guided setup</button>
		</div>
	</div>

	<!-- ══════════════════════════════════════════════════════════════════
	     SCRIPTS
	     ══════════════════════════════════════════════════════════════════ -->
	<script nonce="${nonce}">
		const vscode = acquireVsCodeApi();

		/* ── Onboarding ── */
		let currentStep = 0;
		const totalSteps = 7;

		function startGuided() {
			document.getElementById('hero').classList.add('hidden');
			document.getElementById('guidedOverlay').classList.add('active');
			vscode.postMessage({ type: 'startGuided' });
			renderStepProgress();
		}

		function skipOnboarding() {
			vscode.postMessage({ type: 'skipOnboarding' });
		}

		function skipStep() {
			vscode.postMessage({ type: 'skipStep' });
		}

		function exitGuided() {
			document.getElementById('hero').classList.remove('hidden');
			document.getElementById('guidedOverlay').classList.remove('active');
			document.getElementById('completeScreen').style.display = 'none';
			document.getElementById('stepCard').style.display = 'block';
			vscode.postMessage({ type: 'exitGuided' });
		}

		function finishOnboarding() {
			vscode.postMessage({ type: 'finishOnboarding' });
		}

		function resetOnboarding() {
			vscode.postMessage({ type: 'startGuidedMode' });
		}

		function renderStepProgress() {
			const container = document.getElementById('stepProgress');
			if (!container) return;
			container.innerHTML = '';
			for (let i = 0; i < totalSteps; i++) {
				const dot = document.createElement('div');
				dot.className = 'step-dot';
				if (i < currentStep) dot.classList.add('complete');
				if (i === currentStep) dot.classList.add('active');
				container.appendChild(dot);
			}
		}

		function updateNarrator(text) {
			const el = document.getElementById('narratorText');
			if (el) el.textContent = text;
		}

		function updateStep(step, index) {
			currentStep = index;
			renderStepProgress();
			const titleEl = document.getElementById('stepTitle');
			const descEl = document.getElementById('stepDescription');
			const statusEl = document.getElementById('stepStatus');
			if (titleEl) titleEl.textContent = step.title;
			if (descEl) descEl.textContent = step.description;
			if (statusEl) {
				statusEl.textContent = step.status.charAt(0).toUpperCase() + step.status.slice(1);
				statusEl.className = 'step-status ' + step.status;
			}
			renderSubsteps(step.substeps);
		}

		function renderSubsteps(substeps) {
			const container = document.getElementById('substeps');
			if (!container) return;
			container.innerHTML = '';
			const iconMap = { pending: '○', running: '◐', success: '✓', error: '✗' };
			substeps.forEach(sub => {
				const div = document.createElement('div');
				div.className = 'substep';
				div.id = 'substep-' + sub.id;
				div.innerHTML =
					'<div class="substep-icon ' + sub.status + '">' + (iconMap[sub.status] || '○') + '</div>' +
					'<div class="substep-label">' + sub.label + '</div>' +
					(sub.result ? '<div class="substep-result">' + sub.result + '</div>' : '');
				container.appendChild(div);
			});
		}

		function updateSubstep(stepId, substep) {
			const el = document.getElementById('substep-' + substep.id);
			if (!el) return;
			const iconMap = { pending: '○', running: '◐', success: '✓', error: '✗' };
			const iconEl = el.querySelector('.substep-icon');
			if (iconEl) {
				iconEl.textContent = iconMap[substep.status] || '○';
				iconEl.className = 'substep-icon ' + substep.status;
			}
			if (substep.result) {
				let resultEl = el.querySelector('.substep-result');
				if (!resultEl) {
					resultEl = document.createElement('div');
					resultEl.className = 'substep-result';
					el.appendChild(resultEl);
				}
				resultEl.textContent = substep.result;
			}
		}

		function showComplete(summary) {
			const stepCard = document.getElementById('stepCard');
			const completeScreen = document.getElementById('completeScreen');
			if (stepCard) stepCard.style.display = 'none';
			if (completeScreen) completeScreen.style.display = 'block';
			const summaryEl = document.getElementById('completeSummary');
			if (summaryEl && summary) summaryEl.textContent = summary;
			document.querySelectorAll('.step-dot').forEach(d => { d.classList.remove('active'); d.classList.add('complete'); });
		}

		/* ── Dashboard — Service Cards ── */
		document.querySelectorAll('.service-card').forEach(card => {
			card.addEventListener('click', () => {
				const cmd = card.dataset.cmd;
				if (cmd) vscode.postMessage({ type: 'runCommand', command: cmd });
			});
		});

		/* ── Dashboard — Main Buttons ── */
		const btnStart = document.getElementById('btnStartServices');
		if (btnStart) btnStart.addEventListener('click', () => {
			vscode.postMessage({ type: 'runCommand', command: 'slate/slate_orchestrator.py start' });
		});

		const btnStatus = document.getElementById('btnFullStatus');
		if (btnStatus) btnStatus.addEventListener('click', () => {
			vscode.postMessage({ type: 'showStatus' });
		});

		const btnRerun = document.getElementById('btnRerunSetup');
		if (btnRerun) btnRerun.addEventListener('click', () => {
			vscode.postMessage({ type: 'startGuidedMode' });
		});

		/* ── Dashboard — Quick Actions ── */
		document.querySelectorAll('.action-btn').forEach(btn => {
			btn.addEventListener('click', () => {
				const action = btn.dataset.action;
				switch (action) {
					case 'chat': vscode.postMessage({ type: 'openChat' }); break;
					case 'workflow': vscode.postMessage({ type: 'runCommand', command: 'slate/slate_workflow_manager.py --status' }); break;
					case 'benchmark': vscode.postMessage({ type: 'runCommand', command: 'slate/slate_benchmark.py' }); break;
					case 'security': vscode.postMessage({ type: 'runCommand', command: 'slate/action_guard.py --scan' }); break;
				}
			});
		});

		/* ── Dashboard — Dev Cycle Ring ── */
		const stageNames = { PLAN: 'Planning', CODE: 'Coding', TEST: 'Testing', DEPLOY: 'Deploying', FEEDBACK: 'Feedback' };

		document.querySelectorAll('.stage-segment').forEach(seg => {
			seg.addEventListener('click', () => {
				const stage = seg.dataset.stage;
				if (stage) vscode.postMessage({ type: 'transitionStage', stage });
			});
		});

		function updateDevCycleRing(data) {
			if (!data) return;
			const currentStage = data.current_stage || 'CODE';
			const progress = data.stage_progress_percent || 0;
			const csEl = document.getElementById('currentStage');
			const snEl = document.getElementById('stageName');
			const spEl = document.getElementById('stageProgress');
			const sbEl = document.getElementById('stageBarFill');
			if (csEl) csEl.textContent = currentStage;
			if (snEl) snEl.textContent = stageNames[currentStage] || currentStage;
			if (spEl) spEl.textContent = progress + '% complete';
			if (sbEl) sbEl.style.width = progress + '%';

			const centerTexts = document.querySelectorAll('#devCycleRing text');
			if (centerTexts.length >= 2) {
				centerTexts[0].textContent = currentStage;
				centerTexts[1].textContent = progress + '%';
			}

			['PLAN','CODE','TEST','DEPLOY','FEEDBACK'].forEach(s => {
				const seg = document.getElementById('seg' + s.charAt(0) + s.slice(1).toLowerCase());
				if (seg) {
					if (s === currentStage) { seg.classList.add('active'); seg.setAttribute('opacity','1'); }
					else { seg.classList.remove('active'); seg.setAttribute('opacity','0.4'); }
				}
			});
		}

		/* ── Dashboard — Learning Toggle ── */
		const learningToggle = document.getElementById('learningToggle');
		let learningActive = false;
		if (learningToggle) {
			learningToggle.addEventListener('click', () => {
				learningActive = !learningActive;
				learningToggle.classList.toggle('active', learningActive);
				vscode.postMessage({ type: 'toggleLearning', active: learningActive });
			});
		}

		function updateLearningStats(data) {
			if (!data) return;
			const lvlEl = document.getElementById('learningLevel');
			const xpEl = document.getElementById('learningXP');
			if (lvlEl) lvlEl.textContent = 'Level ' + (data.level || 1);
			if (xpEl) xpEl.textContent = (data.total_xp || 0).toLocaleString() + ' XP';
		}

		/* ── Dashboard — iframe Controls ── */
		const btnRefresh = document.getElementById('btnRefreshFrame');
		if (btnRefresh) btnRefresh.addEventListener('click', () => {
			const frame = document.getElementById('dashboardFrame');
			if (frame) frame.src = frame.src;
		});

		const btnExpand = document.getElementById('btnExpandFrame');
		if (btnExpand) btnExpand.addEventListener('click', () => {
			vscode.postMessage({ type: 'openPanel' });
		});

		const btnExternal = document.getElementById('btnExternalFrame');
		if (btnExternal) btnExternal.addEventListener('click', () => {
			vscode.postMessage({ type: 'openExternal' });
		});

		function retryDashboard() {
			const frame = document.getElementById('dashboardFrame');
			const notice = document.getElementById('offlineNotice');
			if (frame) { frame.style.display = 'block'; frame.src = '${DASHBOARD_URL}'; }
			if (notice) notice.classList.remove('visible');
		}

		/* ── Dashboard — iframe offline detection ── */
		const dashFrame = document.getElementById('dashboardFrame');
		if (dashFrame) {
			dashFrame.addEventListener('error', () => {
				dashFrame.style.display = 'none';
				const notice = document.getElementById('offlineNotice');
				if (notice) notice.classList.add('visible');
			});
		}

		/* ── Message Handler ── */
		window.addEventListener('message', event => {
			const msg = event.data;
			switch (msg.type) {
				case 'stepUpdate': updateStep(msg.step, msg.currentIndex); break;
				case 'narration': updateNarrator(msg.text); break;
				case 'substepUpdate': updateSubstep(msg.stepId, msg.substep); break;
				case 'stepComplete':
					updateStep(msg.step, msg.currentIndex);
					if (msg.step.id === 'complete') {
						const summary = msg.step.substeps.find(s => s.id === 'summary');
						showComplete(summary && summary.result);
					}
					break;
				case 'devCycleUpdate': updateDevCycleRing(msg.data); break;
				case 'learningUpdate': updateLearningStats(msg.data); break;
				case 'interactiveStatus':
					if (msg.data && msg.data.dev_cycle) updateDevCycleRing(msg.data.dev_cycle);
					if (msg.data && msg.data.learning) updateLearningStats(msg.data.learning);
					break;
				case 'statusUpdate':
					document.querySelectorAll('.service-card').forEach(c => {
						c.style.animation = 'slideIn 0.3s ease-out';
						setTimeout(() => c.style.animation = '', 300);
					});
					break;
				case 'guidedExit': exitGuided(); break;
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
