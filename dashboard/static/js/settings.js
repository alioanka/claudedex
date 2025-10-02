// Settings Management

class SettingsManager {
    constructor() {
        this.currentSettings = {};
        this.originalSettings = {};
        this.hasUnsavedChanges = false;
        this.init();
    }

    init() {
        this.loadSettings();
        this.setupEventListeners();
        this.loadSettingsHistory();
    }

    setupEventListeners() {
        // Save button
        const saveBtn = document.getElementById('saveSettings');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveSettings());
        }

        // Revert button
        const revertBtn = document.getElementById('revertSettings');
        if (revertBtn) {
            revertBtn.addEventListener('click', () => this.revertSettings());
        }

        // Input change detection
        const inputs = document.querySelectorAll('.settings-input');
        inputs.forEach(input => {
            input.addEventListener('change', () => {
                this.hasUnsavedChanges = true;
                this.updateSaveButton();
            });
        });

        // Category tabs
        const categoryTabs = document.querySelectorAll('.category-tab');
        categoryTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchCategory(tab.dataset.category);
            });
        });

        // Strategy parameter forms
        document.querySelectorAll('.strategy-form').forEach(form => {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.saveStrategyParams(form);
            });
        });
    }

    async loadSettings() {
        showLoading('Loading settings...');
        
        try {
            const response = await fetch('/api/settings');
            const data = await response.json();
            
            if (data.success) {
                this.currentSettings = data.settings;
                this.originalSettings = JSON.parse(JSON.stringify(data.settings));
                this.populateSettings();
            } else {
                showError('Failed to load settings');
            }
        } catch (error) {
            showError('Error loading settings: ' + error.message);
        } finally {
            hideLoading();
        }
    }

    populateSettings() {
        // Populate trading settings
        if (this.currentSettings.trading) {
            this.setInputValue('maxPositionSize', this.currentSettings.trading.max_position_size);
            this.setInputValue('maxOpenPositions', this.currentSettings.trading.max_open_positions);
            this.setInputValue('defaultStopLoss', this.currentSettings.trading.default_stop_loss);
            this.setInputValue('defaultTakeProfit', this.currentSettings.trading.default_take_profit);
            this.setInputValue('slippageTolerance', this.currentSettings.trading.slippage_tolerance);
            this.setInputValue('tradingEnabled', this.currentSettings.trading.enabled);
        }

        // Populate risk management settings
        if (this.currentSettings.risk_management) {
            this.setInputValue('maxDrawdown', this.currentSettings.risk_management.max_drawdown);
            this.setInputValue('maxDailyLoss', this.currentSettings.risk_management.max_daily_loss);
            this.setInputValue('positionSizeMethod', this.currentSettings.risk_management.position_size_method);
            this.setInputValue('riskPerTrade', this.currentSettings.risk_management.risk_per_trade);
        }

        // Populate API settings
        if (this.currentSettings.api) {
            this.setInputValue('dexscreenerKey', this.currentSettings.api.dexscreener_api_key);
            this.setInputValue('web3Provider', this.currentSettings.api.web3_provider_url);
            this.setInputValue('rpcTimeout', this.currentSettings.api.rpc_timeout);
            this.setInputValue('maxRetries', this.currentSettings.api.max_retries);
        }

        // Populate notification settings
        if (this.currentSettings.notifications) {
            this.setInputValue('telegramEnabled', this.currentSettings.notifications.telegram?.enabled);
            this.setInputValue('telegramBotToken', this.currentSettings.notifications.telegram?.bot_token);
            this.setInputValue('telegramChatId', this.currentSettings.notifications.telegram?.chat_id);
            this.setInputValue('discordEnabled', this.currentSettings.notifications.discord?.enabled);
            this.setInputValue('discordWebhook', this.currentSettings.notifications.discord?.webhook_url);
            this.setInputValue('emailEnabled', this.currentSettings.notifications.email?.enabled);
        }

        this.hasUnsavedChanges = false;
        this.updateSaveButton();
    }

    setInputValue(id, value) {
        const element = document.getElementById(id);
        if (!element) return;

        if (element.type === 'checkbox') {
            element.checked = value;
        } else {
            element.value = value;
        }
    }

    getInputValue(id) {
        const element = document.getElementById(id);
        if (!element) return null;

        if (element.type === 'checkbox') {
            return element.checked;
        } else if (element.type === 'number') {
            return parseFloat(element.value);
        }
        return element.value;
    }

    async saveSettings() {
        if (!this.hasUnsavedChanges) {
            showInfo('No changes to save');
            return;
        }

        // Collect all settings from form
        const updates = {
            trading: {
                max_position_size: this.getInputValue('maxPositionSize'),
                max_open_positions: this.getInputValue('maxOpenPositions'),
                default_stop_loss: this.getInputValue('defaultStopLoss'),
                default_take_profit: this.getInputValue('defaultTakeProfit'),
                slippage_tolerance: this.getInputValue('slippageTolerance'),
                enabled: this.getInputValue('tradingEnabled')
            },
            risk_management: {
                max_drawdown: this.getInputValue('maxDrawdown'),
                max_daily_loss: this.getInputValue('maxDailyLoss'),
                position_size_method: this.getInputValue('positionSizeMethod'),
                risk_per_trade: this.getInputValue('riskPerTrade')
            },
            api: {
                dexscreener_api_key: this.getInputValue('dexscreenerKey'),
                web3_provider_url: this.getInputValue('web3Provider'),
                rpc_timeout: this.getInputValue('rpcTimeout'),
                max_retries: this.getInputValue('maxRetries')
            },
            notifications: {
                telegram: {
                    enabled: this.getInputValue('telegramEnabled'),
                    bot_token: this.getInputValue('telegramBotToken'),
                    chat_id: this.getInputValue('telegramChatId')
                },
                discord: {
                    enabled: this.getInputValue('discordEnabled'),
                    webhook_url: this.getInputValue('discordWebhook')
                },
                email: {
                    enabled: this.getInputValue('emailEnabled')
                }
            }
        };

        showLoading('Saving settings...');

        try {
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    updates: updates,
                    reason: 'Manual update via dashboard'
                })
            });

            const data = await response.json();

            if (data.success) {
                showSuccess('Settings saved successfully');
                this.currentSettings = data.settings;
                this.originalSettings = JSON.parse(JSON.stringify(data.settings));
                this.hasUnsavedChanges = false;
                this.updateSaveButton();
                this.loadSettingsHistory();
            } else {
                showError('Failed to save settings: ' + data.error);
            }
        } catch (error) {
            showError('Error saving settings: ' + error.message);
        } finally {
            hideLoading();
        }
    }

    revertSettings() {
        if (!this.hasUnsavedChanges) {
            showInfo('No changes to revert');
            return;
        }

        if (confirm('Are you sure you want to revert all changes?')) {
            this.currentSettings = JSON.parse(JSON.stringify(this.originalSettings));
            this.populateSettings();
            showSuccess('Changes reverted');
        }
    }

    async loadSettingsHistory() {
        try {
            const response = await fetch('/api/settings/history?limit=10');
            const data = await response.json();

            if (data.success) {
                this.displaySettingsHistory(data.history);
            }
        } catch (error) {
            console.error('Error loading settings history:', error);
        }
    }

    displaySettingsHistory(history) {
        const container = document.getElementById('settingsHistory');
        if (!container) return;

        container.innerHTML = history.map(item => `
            <div class="history-item">
                <div class="history-header">
                    <span class="history-time">${formatTimestamp(item.timestamp)}</span>
                    <span class="history-user">${item.user || 'System'}</span>
                </div>
                <div class="history-reason">${item.reason}</div>
                <button class="btn btn-sm btn-secondary" onclick="settingsManager.revertToVersion('${item.version}')">
                    Restore
                </button>
            </div>
        `).join('');
    }

    async revertToVersion(version) {
        if (!confirm('Are you sure you want to restore this version?')) {
            return;
        }

        showLoading('Restoring settings...');

        try {
            const response = await fetch('/api/settings/revert', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ version: version })
            });

            const data = await response.json();

            if (data.success) {
                showSuccess('Settings restored successfully');
                await this.loadSettings();
            } else {
                showError('Failed to restore settings: ' + data.error);
            }
        } catch (error) {
            showError('Error restoring settings: ' + error.message);
        } finally {
            hideLoading();
        }
    }

    switchCategory(category) {
        // Update active tab
        document.querySelectorAll('.category-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-category="${category}"]`).classList.add('active');

        // Show/hide sections
        document.querySelectorAll('.settings-section').forEach(section => {
            section.style.display = 'none';
        });
        document.getElementById(category + 'Settings').style.display = 'block';
    }

    async saveStrategyParams(form) {
        const strategy = form.dataset.strategy;
        const formData = new FormData(form);
        const params = Object.fromEntries(formData.entries());

        showLoading('Saving strategy parameters...');

        try {
            const response = await fetch(`/api/strategies/${strategy}/params`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            const data = await response.json();

            if (data.success) {
                showSuccess(`${strategy} parameters saved`);
            } else {
                showError('Failed to save parameters: ' + data.error);
            }
        } catch (error) {
            showError('Error saving parameters: ' + error.message);
        } finally {
            hideLoading();
        }
    }

    updateSaveButton() {
        const saveBtn = document.getElementById('saveSettings');
        if (saveBtn) {
            saveBtn.disabled = !this.hasUnsavedChanges;
            saveBtn.textContent = this.hasUnsavedChanges ? 'Save Changes' : 'No Changes';
        }
    }

    async testNotification(channel) {
        showLoading(`Testing ${channel} notification...`);

        try {
            const response = await fetch(`/api/notifications/test/${channel}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                showSuccess(`Test notification sent to ${channel}`);
            } else {
                showError(`Failed to send test notification: ${data.error}`);
            }
        } catch (error) {
            showError(`Error testing notification: ${error.message}`);
        } finally {
            hideLoading();
        }
    }

    exportSettings() {
        const dataStr = JSON.stringify(this.currentSettings, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `bot-settings-${Date.now()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }

    async importSettings(file) {
        try {
            const text = await file.text();
            const settings = JSON.parse(text);
            
            if (confirm('This will replace all current settings. Continue?')) {
                this.currentSettings = settings;
                this.populateSettings();
                this.hasUnsavedChanges = true;
                this.updateSaveButton();
                showSuccess('Settings imported. Click Save to apply.');
            }
        } catch (error) {
            showError('Invalid settings file: ' + error.message);
        }
    }
}

// Initialize settings manager
const settingsManager = new SettingsManager();

// Export for use in HTML
window.settingsManager = settingsManager;