// Modified: 2026-02-07T22:00:00Z | Author: COPILOT | Change: Fix full-panel dashboard theme to SLATE M3 ProArt + unified view cleanup
import * as vscode from 'vscode';
import { registerSlateParticipant } from './slateParticipant';
import { registerSlateTools } from './tools';
import { SlateUnifiedDashboardViewProvider } from './slateUnifiedDashboardView';
import { registerServiceMonitor } from './slateServiceMonitor';
import {
	applySchematicBackground,
	registerBackgroundCommands,
	watchForStateChanges
} from './slateSchematicBackground';

const DASHBOARD_URL = 'http://127.0.0.1:8080';
const SLATE_THEME_ID = 'SLATE Dark';

/** Status bar item showing SLATE is installed */
let slateStatusBarItem: vscode.StatusBarItem;

export function activate(context: vscode.ExtensionContext) {
	registerSlateTools(context);
	registerSlateParticipant(context);

	// ─── SLATE Theme & Background Initialization ───────────────────────────
	// Embodies the SLATE ethos: systems evolve with progress
	initializeSlateEnvironment(context);

	// Create SLATE status bar indicator
	slateStatusBarItem = vscode.window.createStatusBarItem(
		vscode.StatusBarAlignment.Left,
		100
	);
	slateStatusBarItem.text = '$(circuit-board) SLATE';
	slateStatusBarItem.tooltip = 'S.L.A.T.E. — Synchronized Living Architecture for Transformation and Evolution\n\nClick to show system status';
	slateStatusBarItem.command = 'slate.showStatus';
	slateStatusBarItem.backgroundColor = undefined;
	slateStatusBarItem.show();
	context.subscriptions.push(slateStatusBarItem);

	// Register service monitor for auto-restart
	const serviceMonitor = registerServiceMonitor(context);

	// Register the unified dashboard (combines guided setup, control board, and dashboard)
	const unifiedDashboardProvider = new SlateUnifiedDashboardViewProvider(context.extensionUri, context);
	context.subscriptions.push(
		vscode.window.registerWebviewViewProvider(
			SlateUnifiedDashboardViewProvider.viewType,
			unifiedDashboardProvider,
			{ webviewOptions: { retainContextWhenHidden: true } }
		)
	);

	// Refresh command
	context.subscriptions.push(
		vscode.commands.registerCommand('slate.refreshDashboard', () => {
			unifiedDashboardProvider.refresh();
		})
	);

	// Register the status command
	context.subscriptions.push(
		vscode.commands.registerCommand('slate.showStatus', async () => {
			const terminal = vscode.window.createTerminal('SLATE Status');
			terminal.show();
			terminal.sendText(`"${getSlateConfig().pythonPath}" slate/slate_status.py --quick`);
		})
	);

	context.subscriptions.push(
		vscode.commands.registerCommand('slate.openDashboard', async () => {
			const panel = vscode.window.createWebviewPanel(
				'slateDashboard',
				'SLATE Dashboard',
				vscode.ViewColumn.One,
				{
					enableScripts: true,
					retainContextWhenHidden: true,
				}
			);

			panel.webview.onDidReceiveMessage((message) => {
				if (message?.type === 'openExternal') {
					void vscode.env.openExternal(vscode.Uri.parse(DASHBOARD_URL));
				}
			});

			panel.webview.html = getDashboardHtml(panel.webview);
		})
	);
}

export function deactivate() { }

// ─── SLATE Environment Initialization ──────────────────────────────────────
// This implements the SLATE ethos: the system evolves with progress

/**
 * Initialize the SLATE visual environment:
 * 1. Apply SLATE Dark theme (on first install or if requested)
 * 2. Generate and apply the evolving schematic background
 * 3. Set up watchers for automatic background evolution
 */
async function initializeSlateEnvironment(context: vscode.ExtensionContext): Promise<void> {
	const isFirstActivation = !context.globalState.get<boolean>('slateInitialized');

	// Register background commands
	registerBackgroundCommands(context);

	// On first activation, offer to apply SLATE theme
	if (isFirstActivation) {
		await context.globalState.update('slateInitialized', true);

		const applyTheme = await vscode.window.showInformationMessage(
			'Welcome to SLATE! Apply the SLATE Dark theme and schematic background?',
			'Yes, Transform VS Code',
			'Just Theme',
			'Skip'
		);

		if (applyTheme === 'Yes, Transform VS Code') {
			// Apply theme
			await vscode.workspace.getConfiguration('workbench').update(
				'colorTheme',
				SLATE_THEME_ID,
				vscode.ConfigurationTarget.Global
			);

			// Enable and generate background
			await vscode.workspace.getConfiguration('slate').update(
				'background.enabled',
				true,
				vscode.ConfigurationTarget.Global
			);
			await applySchematicBackground(context);

			// Set up watchers for evolution
			watchForStateChanges(context);

			vscode.window.showInformationMessage(
				'SLATE environment activated! Your VS Code will evolve as your system progresses.'
			);
		} else if (applyTheme === 'Just Theme') {
			await vscode.workspace.getConfiguration('workbench').update(
				'colorTheme',
				SLATE_THEME_ID,
				vscode.ConfigurationTarget.Global
			);
		}
	} else {
		// On subsequent activations, refresh background if enabled
		const config = vscode.workspace.getConfiguration('slate');
		if (config.get<boolean>('background.enabled', false)) {
			// Generate background silently
			await applySchematicBackground(context);
			watchForStateChanges(context);
		}
	}

	// Register apply theme command
	context.subscriptions.push(
		vscode.commands.registerCommand('slate.applyTheme', async () => {
			await vscode.workspace.getConfiguration('workbench').update(
				'colorTheme',
				SLATE_THEME_ID,
				vscode.ConfigurationTarget.Global
			);

			const enableBg = await vscode.window.showInformationMessage(
				'SLATE Dark theme applied! Enable evolving schematic background?',
				'Yes',
				'No'
			);

			if (enableBg === 'Yes') {
				await vscode.workspace.getConfiguration('slate').update(
					'background.enabled',
					true,
					vscode.ConfigurationTarget.Global
				);
				await applySchematicBackground(context);
				watchForStateChanges(context);
			}
		})
	);
}

/** SLATE workspace configuration */
export interface SlateConfig {
	pythonPath: string;
	workspacePath: string;
}

/** Get SLATE configuration from workspace */
export function getSlateConfig(): SlateConfig {
	const workspaceFolders = vscode.workspace.workspaceFolders;
	const workspacePath = workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();

	return {
		pythonPath: `${workspacePath}\\.venv\\Scripts\\python.exe`,
		workspacePath,
	};
}

function getDashboardHtml(webview: vscode.Webview): string {
	const nonce = getNonce();
	const csp = [
		"default-src 'none'",
		`frame-src ${DASHBOARD_URL}`,
		`img-src ${DASHBOARD_URL} data:`,
		"style-src 'unsafe-inline'",
		`script-src 'nonce-${nonce}'`,
	].join('; ');

	return `<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1.0" />
	<meta http-equiv="Content-Security-Policy" content="${csp}" />
	<title>SLATE Dashboard</title>
	<style>
		html, body {
			height: 100%;
			width: 100%;
			margin: 0;
			padding: 0;
			background: #0a0a0a;
			color: #F5F0EB;
			font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
		}
		.toolbar {
			display: flex;
			align-items: center;
			justify-content: space-between;
			padding: 10px 14px;
			border-bottom: 1px solid rgba(255,255,255,0.08);
			background: #050505;
		}
		.title {
			font-size: 14px;
			letter-spacing: 0.08em;
			text-transform: uppercase;
			color: #B87333;
			font-weight: 600;
		}
		.actions {
			display: flex;
			gap: 8px;
		}
		button {
			background: #111111;
			color: #F5F0EB;
			border: 1px solid rgba(255,255,255,0.12);
			border-radius: 8px;
			padding: 6px 14px;
			cursor: pointer;
			font-size: 12px;
			font-family: inherit;
			transition: all 0.2s;
		}
		button:hover {
			background: #1a1a1a;
			border-color: #B87333;
			color: #C9956B;
		}
		.frame {
			height: calc(100% - 44px);
			width: 100%;
			border: none;
			background: #0a0a0a;
		}
	</style>
</head>
<body>
	<div class="toolbar">
		<div class="title">SLATE Dashboard (127.0.0.1:8080)</div>
		<div class="actions">
			<button id="openExternal">Open in Browser</button>
		</div>
	</div>
	<iframe class="frame" src="${DASHBOARD_URL}" title="SLATE Dashboard"></iframe>
	<script nonce="${nonce}">
		const vscode = acquireVsCodeApi();
		document.getElementById('openExternal').addEventListener('click', () => {
			vscode.postMessage({ type: 'openExternal' });
		});
	</script>
</body>
</html>`;
}

function getNonce(): string {
	const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
	let text = '';
	for (let i = 0; i < 32; i += 1) {
		text += possible.charAt(Math.floor(Math.random() * possible.length));
	}
	return text;
}
