// Modified: 2026-02-07T06:00:00Z | Author: Claude | Change: Add service monitor for auto-restart of SLATE services
import * as vscode from 'vscode';
import * as http from 'http';
import * as cp from 'child_process';
import { getSlateConfig } from './extension';

const DASHBOARD_URL = 'http://127.0.0.1:8080/health';
const HEALTH_CHECK_INTERVAL = 15000; // 15 seconds
const MAX_RESTART_ATTEMPTS = 3;
const RESTART_COOLDOWN = 60000; // 1 minute

interface ServiceState {
	running: boolean;
	lastCheck: number;
	restartAttempts: number;
	lastRestart: number;
}

/**
 * Service Monitor for SLATE Dashboard
 * Automatically checks if dashboard is running and restarts it if needed.
 */
export class SlateServiceMonitor implements vscode.Disposable {
	private _interval: NodeJS.Timeout | undefined;
	private _state: ServiceState = {
		running: false,
		lastCheck: 0,
		restartAttempts: 0,
		lastRestart: 0,
	};
	private _statusBarItem: vscode.StatusBarItem;
	private _outputChannel: vscode.OutputChannel;
	private _dashboardProcess: cp.ChildProcess | undefined;

	constructor() {
		this._statusBarItem = vscode.window.createStatusBarItem(
			vscode.StatusBarAlignment.Right,
			100
		);
		this._statusBarItem.command = 'slate.toggleService';
		this._outputChannel = vscode.window.createOutputChannel('SLATE Services');
	}

	/**
	 * Start the service monitor
	 */
	public start(): void {
		this._log('Service monitor started');
		this._updateStatusBar('checking');

		// Initial check
		void this._checkHealth();

		// Start periodic checks
		this._interval = setInterval(() => {
			void this._checkHealth();
		}, HEALTH_CHECK_INTERVAL);

		this._statusBarItem.show();
	}

	/**
	 * Stop the service monitor
	 */
	public stop(): void {
		if (this._interval) {
			clearInterval(this._interval);
			this._interval = undefined;
		}
		this._statusBarItem.hide();
		this._log('Service monitor stopped');
	}

	/**
	 * Check dashboard health
	 */
	private async _checkHealth(): Promise<void> {
		const now = Date.now();
		this._state.lastCheck = now;

		try {
			const healthy = await this._pingDashboard();
			this._state.running = healthy;

			if (healthy) {
				// Reset restart counter on success
				if (now - this._state.lastRestart > RESTART_COOLDOWN) {
					this._state.restartAttempts = 0;
				}
				this._updateStatusBar('running');
			} else {
				this._updateStatusBar('stopped');
				// Attempt auto-restart
				await this._attemptRestart();
			}
		} catch {
			this._state.running = false;
			this._updateStatusBar('error');
		}
	}

	/**
	 * Ping the dashboard health endpoint
	 */
	private _pingDashboard(): Promise<boolean> {
		return new Promise((resolve) => {
			const req = http.get(DASHBOARD_URL, { timeout: 5000 }, (res) => {
				resolve(res.statusCode === 200);
			});
			req.on('error', () => resolve(false));
			req.on('timeout', () => {
				req.destroy();
				resolve(false);
			});
		});
	}

	/**
	 * Attempt to restart the dashboard server
	 */
	private async _attemptRestart(): Promise<void> {
		const now = Date.now();

		// Check cooldown and max attempts
		if (this._state.restartAttempts >= MAX_RESTART_ATTEMPTS) {
			if (now - this._state.lastRestart < RESTART_COOLDOWN * 5) {
				this._log('Max restart attempts reached, waiting for cooldown...');
				return;
			}
			// Reset after extended cooldown
			this._state.restartAttempts = 0;
		}

		// Exponential backoff
		const backoff = Math.pow(2, this._state.restartAttempts) * 5000;
		if (now - this._state.lastRestart < backoff) {
			return;
		}

		this._state.restartAttempts++;
		this._state.lastRestart = now;

		this._log(`Attempting restart (${this._state.restartAttempts}/${MAX_RESTART_ATTEMPTS})...`);
		this._updateStatusBar('restarting');

		try {
			await this._startDashboard();

			// Wait and verify
			await new Promise(resolve => setTimeout(resolve, 3000));
			const running = await this._pingDashboard();

			if (running) {
				this._log('Dashboard restarted successfully');
				this._state.running = true;
				this._updateStatusBar('running');
				void vscode.window.showInformationMessage('SLATE Dashboard auto-restarted');
			} else {
				this._log('Dashboard restart failed - not responding');
				this._updateStatusBar('error');
			}
		} catch (err) {
			this._log(`Restart error: ${err}`);
			this._updateStatusBar('error');
		}
	}

	/**
	 * Start the dashboard server process
	 */
	private _startDashboard(): Promise<void> {
		return new Promise((resolve, reject) => {
			const config = getSlateConfig();
			const dashboardScript = `${config.workspacePath}\\agents\\slate_dashboard_server.py`;

			// Kill existing process if any
			if (this._dashboardProcess && !this._dashboardProcess.killed) {
				this._dashboardProcess.kill();
			}

			try {
				this._dashboardProcess = cp.spawn(
					config.pythonPath,
					[dashboardScript],
					{
						cwd: config.workspacePath,
						env: {
							...process.env,
							PYTHONPATH: config.workspacePath,
							SLATE_WORKSPACE: config.workspacePath,
						},
						detached: true,
						stdio: 'ignore',
						windowsHide: true,
					}
				);

				this._dashboardProcess.unref();
				this._log(`Dashboard process started (PID: ${this._dashboardProcess.pid})`);
				resolve();
			} catch (err) {
				reject(err);
			}
		});
	}

	/**
	 * Toggle service (start if stopped, show status if running)
	 */
	public async toggleService(): Promise<void> {
		if (this._state.running) {
			const action = await vscode.window.showInformationMessage(
				'SLATE Dashboard is running on http://127.0.0.1:8080',
				'Open Dashboard',
				'Restart',
				'Stop'
			);

			if (action === 'Open Dashboard') {
				void vscode.env.openExternal(vscode.Uri.parse('http://127.0.0.1:8080'));
			} else if (action === 'Restart') {
				this._state.restartAttempts = 0;
				await this._startDashboard();
				void vscode.window.showInformationMessage('SLATE Dashboard restarting...');
			} else if (action === 'Stop') {
				if (this._dashboardProcess) {
					this._dashboardProcess.kill();
				}
				this._state.running = false;
				this._updateStatusBar('stopped');
			}
		} else {
			const action = await vscode.window.showWarningMessage(
				'SLATE Dashboard is not running',
				'Start Dashboard',
				'Start with Watchdog'
			);

			if (action === 'Start Dashboard') {
				await this._startDashboard();
			} else if (action === 'Start with Watchdog') {
				await this._startWatchdog();
			}
		}
	}

	/**
	 * Start the watchdog service
	 */
	private async _startWatchdog(): Promise<void> {
		const config = getSlateConfig();
		const watchdogScript = `${config.workspacePath}\\slate\\slate_service_watchdog.py`;

		const terminal = vscode.window.createTerminal('SLATE Watchdog');
		terminal.show();
		terminal.sendText(`"${config.pythonPath}" "${watchdogScript}" start`);
	}

	/**
	 * Force restart the dashboard
	 */
	public async forceRestart(): Promise<void> {
		this._log('Force restart requested');
		this._state.restartAttempts = 0;
		await this._startDashboard();

		// Wait and check
		await new Promise(resolve => setTimeout(resolve, 3000));
		await this._checkHealth();
	}

	/**
	 * Update status bar item
	 */
	private _updateStatusBar(status: 'running' | 'stopped' | 'checking' | 'restarting' | 'error'): void {
		switch (status) {
			case 'running':
				this._statusBarItem.text = '$(check) SLATE';
				this._statusBarItem.tooltip = 'SLATE Dashboard running - Click for options';
				this._statusBarItem.backgroundColor = undefined;
				break;
			case 'stopped':
				this._statusBarItem.text = '$(circle-slash) SLATE';
				this._statusBarItem.tooltip = 'SLATE Dashboard stopped - Click to start';
				this._statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
				break;
			case 'checking':
				this._statusBarItem.text = '$(sync~spin) SLATE';
				this._statusBarItem.tooltip = 'Checking SLATE Dashboard...';
				this._statusBarItem.backgroundColor = undefined;
				break;
			case 'restarting':
				this._statusBarItem.text = '$(sync~spin) SLATE';
				this._statusBarItem.tooltip = 'Restarting SLATE Dashboard...';
				this._statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
				break;
			case 'error':
				this._statusBarItem.text = '$(error) SLATE';
				this._statusBarItem.tooltip = 'SLATE Dashboard error - Click to restart';
				this._statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
				break;
		}
	}

	private _log(message: string): void {
		const timestamp = new Date().toISOString();
		this._outputChannel.appendLine(`[${timestamp}] ${message}`);
	}

	public dispose(): void {
		this.stop();
		this._statusBarItem.dispose();
		this._outputChannel.dispose();
		if (this._dashboardProcess && !this._dashboardProcess.killed) {
			this._dashboardProcess.kill();
		}
	}
}

/**
 * Create and register the service monitor
 */
export function registerServiceMonitor(context: vscode.ExtensionContext): SlateServiceMonitor {
	const monitor = new SlateServiceMonitor();

	context.subscriptions.push(monitor);

	// Register commands
	context.subscriptions.push(
		vscode.commands.registerCommand('slate.toggleService', () => {
			void monitor.toggleService();
		})
	);

	context.subscriptions.push(
		vscode.commands.registerCommand('slate.restartDashboard', () => {
			void monitor.forceRestart();
		})
	);

	// Auto-start the monitor
	monitor.start();

	return monitor;
}
