// Modified: 2026-02-07T04:57:00Z | Author: COPILOT | Change: Add dashboard webview command
import * as vscode from 'vscode';
import { registerSlateParticipant } from './slateParticipant';
import { registerSlateTools } from './tools';

const DASHBOARD_URL = 'http://127.0.0.1:8080';

export function activate(context: vscode.ExtensionContext) {
	registerSlateTools(context);
	registerSlateParticipant(context);

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
			background: #0b0c10;
			color: #e5e7eb;
			font-family: "Segoe UI", Arial, sans-serif;
		}
		.toolbar {
			display: flex;
			align-items: center;
			justify-content: space-between;
			padding: 10px 14px;
			border-bottom: 1px solid #1f2937;
			background: #0f1115;
		}
		.title {
			font-size: 14px;
			letter-spacing: 0.04em;
			text-transform: uppercase;
			color: #9ca3af;
		}
		.actions {
			display: flex;
			gap: 8px;
		}
		button {
			background: #1f2937;
			color: #e5e7eb;
			border: 1px solid #374151;
			border-radius: 6px;
			padding: 6px 10px;
			cursor: pointer;
			font-size: 12px;
		}
		button:hover {
			background: #374151;
		}
		.frame {
			height: calc(100% - 44px);
			width: 100%;
			border: none;
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
