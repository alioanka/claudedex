// dashboard/static/js/settings.js

document.addEventListener('DOMContentLoaded', () => {
    const settingsManager = new SettingsManager();
    window.settingsManager = settingsManager;
});

class SettingsManager {
    constructor() {
        this.settings = {};
        this.initialSettings = {};
        this.bindEvents();
        this.loadSettings();
    }

    bindEvents() {
        // Tab switching
        document.querySelectorAll('.category-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e));
        });

        // Save button
        const saveButton = document.getElementById('saveAllSettings');
        if (saveButton) {
            saveButton.addEventListener('click', () => this.saveAllSettings());
        }

        // Password change form
        const passwordForm = document.getElementById('changePasswordForm');
        if (passwordForm) {
            console.log('Password form found, attaching event listener');
            passwordForm.addEventListener('submit', (e) => {
                console.log('Password form submitted');
                e.preventDefault();
                e.stopPropagation();
                this.handlePasswordChange(e);
                return false;
            });
        } else {
            console.error('Password form not found in DOM');
        }

        // Sensitive config management
        const addSensitiveBtn = document.getElementById('addSensitiveConfigBtn');
        if (addSensitiveBtn) {
            addSensitiveBtn.addEventListener('click', () => this.showSensitiveConfigModal());
        }

        const closeSensitiveModal = document.getElementById('closeSensitiveConfigModal');
        if (closeSensitiveModal) {
            closeSensitiveModal.addEventListener('click', () => this.hideSensitiveConfigModal());
        }

        const cancelSensitiveBtn = document.getElementById('cancelSensitiveConfigBtn');
        if (cancelSensitiveBtn) {
            cancelSensitiveBtn.addEventListener('click', () => this.hideSensitiveConfigModal());
        }

        const saveSensitiveBtn = document.getElementById('saveSensitiveConfigBtn');
        if (saveSensitiveBtn) {
            saveSensitiveBtn.addEventListener('click', () => this.saveSensitiveConfig());
        }
    }

    switchTab(event) {
        event.preventDefault();
        const clickedTab = event.currentTarget;
        const category = clickedTab.dataset.category;

        document.querySelectorAll('.category-tab').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.settings-section').forEach(section => {
            section.style.display = 'none';
        });

        clickedTab.classList.add('active');
        const sectionId = `${category}-settings`;
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = 'block';
        }

        // Hide/show save buttons based on tab
        const pageActions = document.querySelector('.page-actions');
        if (pageActions) {
            if (category === 'account' || category === 'sensitive') {
                pageActions.style.display = 'none';
            } else {
                pageActions.style.display = 'flex';
            }
        }

        // Load sensitive configs when tab is opened
        if (category === 'sensitive') {
            this.loadSensitiveConfigs();
        }
    }

    async loadSettings() {
        try {
            const response = await apiGet('/api/settings/all');
            if (response.success && response.data) {
                this.settings = response.data;
                this.initialSettings = JSON.parse(JSON.stringify(response.data)); // Deep copy
                this.renderAllForms(this.settings);
                // Activate the first tab by default
                document.querySelector('.category-tab').click();
                showToast('success', 'Settings loaded successfully');
            } else {
                showToast('error', response.error || 'Failed to load settings.');
            }
        } catch (error) {
            showToast('error', `Error loading settings: ${error.message}`);
        }
    }

    renderAllForms(data) {
        for (const category in data) {
            const container = document.getElementById(`${category}-settings-form`);
            if (container) {
                container.innerHTML = ''; // Clear previous content
                const formContent = this.renderFormCategory(category, data[category]);
                container.appendChild(formContent);
            }
        }
    }

    renderFormCategory(category, categoryData) {
        const fragment = document.createDocumentFragment();
        for (const key in categoryData) {
            const setting = categoryData[key];
            const settingElement = this.createSettingElement(category, key, setting);
            if (settingElement) {
                fragment.appendChild(settingElement);
            }
        }
        return fragment;
    }

    createSettingElement(category, key, setting) {
        const formGroup = document.createElement('div');
        formGroup.className = 'form-group';

        // Extract value and metadata
        const value = setting.value !== undefined ? setting.value : setting;
        const description = setting.description || '';
        const requiresRestart = setting.requires_restart || false;
        const valueType = setting.value_type || typeof value;

        // Create label
        const label = document.createElement('label');
        label.className = 'form-label';
        label.setAttribute('for', `${category}-${key}`);
        label.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

        if (requiresRestart) {
            const restartBadge = document.createElement('span');
            restartBadge.className = 'badge badge-warning';
            restartBadge.textContent = 'Requires Restart';
            restartBadge.style.marginLeft = '8px';
            label.appendChild(restartBadge);
        }

        // Create input based on type
        let input;
        if (valueType === 'bool' || typeof value === 'boolean') {
            const wrapper = document.createElement('div');
            wrapper.className = 'checkbox-wrapper';

            input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = value;
            input.className = 'form-checkbox';
            input.id = `${category}-${key}`;

            const checkboxLabel = document.createElement('label');
            checkboxLabel.setAttribute('for', `${category}-${key}`);
            checkboxLabel.className = 'checkbox-label';
            checkboxLabel.textContent = description || 'Enable';

            wrapper.appendChild(input);
            wrapper.appendChild(checkboxLabel);
            formGroup.appendChild(label);
            formGroup.appendChild(wrapper);
        } else {
            if (valueType === 'int' || valueType === 'float' || typeof value === 'number') {
                input = document.createElement('input');
                input.type = 'number';
                input.step = valueType === 'float' ? '0.01' : '1';
                input.value = value;
                input.className = 'form-control';
            } else if (valueType === 'string' || typeof value === 'string') {
                input = document.createElement('input');
                input.type = 'text';
                input.value = value;
                input.className = 'form-control';
            } else if (valueType === 'json') {
                input = document.createElement('textarea');
                input.value = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
                input.className = 'form-control';
                input.rows = 4;
            } else {
                // Fallback for unknown types
                input = document.createElement('textarea');
                input.value = JSON.stringify(value, null, 2);
                input.className = 'form-control';
                input.rows = 4;
                input.readOnly = true;
            }

            input.id = `${category}-${key}`;

            formGroup.appendChild(label);
            formGroup.appendChild(input);
        }

        input.dataset.category = category;
        input.dataset.key = key;
        input.dataset.valueType = valueType;

        // Add description as help text
        if (description) {
            const helpText = document.createElement('small');
            helpText.className = 'form-text text-muted';
            helpText.textContent = description;
            formGroup.appendChild(helpText);
        }

        return formGroup;
    }

    async saveAllSettings() {
        showToast('info', 'Saving all settings...');
        let allSucceeded = true;

        for (const category in this.settings) {
            const container = document.getElementById(`${category}-settings-form`);
            if (!container) continue;

            const updates = {};
            const inputs = container.querySelectorAll('[data-key]');

            inputs.forEach(input => {
                const key = input.dataset.key;
                const valueType = input.dataset.valueType;

                let value;
                if (input.type === 'checkbox') {
                    value = input.checked;
                } else if (input.type === 'number') {
                    value = parseFloat(input.value);
                } else if (input.tagName === 'TEXTAREA' && input.readOnly) {
                    // Skip read-only textareas
                    return;
                } else if (valueType === 'json') {
                    try {
                        value = JSON.parse(input.value);
                    } catch (e) {
                        value = input.value;
                    }
                } else {
                    value = input.value;
                }

                // Compare with initial value (handle nested structure)
                const initialValue = this.initialSettings[category]?.[key]?.value !== undefined
                    ? this.initialSettings[category][key].value
                    : this.initialSettings[category]?.[key];

                // Only include changed values
                if (initialValue !== value) {
                    updates[key] = value;
                }
            });

            if (Object.keys(updates).length > 0) {
                const success = await this.sendUpdateRequest(category, updates);
                if (!success) {
                    allSucceeded = false;
                }
            }
        }

        if (allSucceeded) {
            showToast('success', 'All changed settings saved successfully!');
            await this.loadSettings(); // Reload to confirm changes
        } else {
            showToast('error', 'One or more settings failed to save. Please review the form.');
        }
    }

    async sendUpdateRequest(category, updates) {
        try {
            const response = await apiPost('/api/settings/update', {
                config_type: category,
                updates: updates,
            });

            if (response.success) {
                return true;
            } else {
                showToast('error', `Failed to update ${category}: ${response.error}`);
                return false;
            }
        } catch (error) {
            showToast('error', `Error saving ${category}: ${error.message}`);
            return false;
        }
    }

    async handlePasswordChange(event) {
        console.log('handlePasswordChange called');
        event.preventDefault();
        event.stopPropagation();

        const currentPassword = document.getElementById('currentPassword').value;
        const newPassword = document.getElementById('newPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;

        console.log('Password change form values:', {
            currentPasswordLength: currentPassword.length,
            newPasswordLength: newPassword.length,
            confirmPasswordLength: confirmPassword.length
        });

        // Validate passwords match
        if (newPassword !== confirmPassword) {
            console.error('Passwords do not match');
            showToast('error', 'New passwords do not match');
            return false;
        }

        // Validate password length
        if (newPassword.length < 8) {
            console.error('Password too short');
            showToast('error', 'New password must be at least 8 characters');
            return false;
        }

        try {
            console.log('Sending password change request...');
            showToast('info', 'Changing password...');

            const response = await apiPost('/api/auth/change-password', {
                old_password: currentPassword,
                new_password: newPassword
            });

            console.log('Password change response:', response);

            if (response.success) {
                showToast('success', 'Password changed successfully! You will be logged out shortly.');

                // Clear form
                document.getElementById('changePasswordForm').reset();

                // Redirect to login after 2 seconds
                setTimeout(() => {
                    window.location.href = '/login';
                }, 2000);
            } else {
                console.error('Password change failed:', response.error);
                showToast('error', response.error || 'Failed to change password');
            }
        } catch (error) {
            console.error('Password change error:', error);
            showToast('error', `Error changing password: ${error.message}`);
        }

        return false;
    }

    // ==================== Sensitive Config Management ====================

    async loadSensitiveConfigs() {
        try {
            const response = await apiGet('/api/settings/sensitive/list');
            if (response.success && response.data) {
                this.renderSensitiveConfigs(response.data);
            } else {
                showToast('error', response.error || 'Failed to load sensitive configs');
                document.getElementById('sensitive-configs-tbody').innerHTML =
                    '<tr><td colspan="5" class="text-center">Failed to load configs</td></tr>';
            }
        } catch (error) {
            showToast('error', `Error loading sensitive configs: ${error.message}`);
            document.getElementById('sensitive-configs-tbody').innerHTML =
                '<tr><td colspan="5" class="text-center">Error loading configs</td></tr>';
        }
    }

    renderSensitiveConfigs(configs) {
        const tbody = document.getElementById('sensitive-configs-tbody');

        if (configs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center">No sensitive configs found</td></tr>';
            return;
        }

        tbody.innerHTML = configs.map(config => {
            const lastRotated = config.last_rotated ?
                new Date(config.last_rotated).toLocaleDateString() : 'Never';

            return `
                <tr>
                    <td><code>${config.key}</code></td>
                    <td>${config.description || '-'}</td>
                    <td>${lastRotated}</td>
                    <td>${config.rotation_interval_days} days</td>
                    <td>
                        <button class="btn btn-sm btn-primary" onclick="settingsManager.editSensitiveConfig('${config.key}')" style="margin-right: 5px;">
                            <i class="fas fa-edit"></i> Edit
                        </button>
                        <button class="btn btn-sm btn-danger" onclick="settingsManager.deleteSensitiveConfig('${config.key}')">
                            <i class="fas fa-trash"></i> Delete
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
    }

    showSensitiveConfigModal(isEdit = false) {
        // Clear form
        document.getElementById('sensitiveConfigKey').value = '';
        document.getElementById('sensitiveConfigValue').value = '';
        document.getElementById('sensitiveConfigDescription').value = '';
        document.getElementById('sensitiveConfigRotation').value = '30';

        // Update modal title
        document.getElementById('sensitiveConfigModalTitle').textContent =
            isEdit ? 'Edit Sensitive Configuration' : 'Add Sensitive Configuration';

        // Disable key field if editing (can't change key)
        document.getElementById('sensitiveConfigKey').readOnly = isEdit;

        // Show modal
        const modal = document.getElementById('sensitiveConfigModal');
        modal.style.display = 'flex';
        // Add active class after a small delay to trigger animation
        setTimeout(() => modal.classList.add('active'), 10);
    }

    async editSensitiveConfig(key) {
        try {
            showToast('info', 'Loading sensitive config...');

            // Fetch the decrypted value from API
            const response = await apiGet(`/api/settings/sensitive/${key}`);

            if (response.success && response.data) {
                const config = response.data;

                // Populate the modal with existing values
                document.getElementById('sensitiveConfigKey').value = config.key;
                document.getElementById('sensitiveConfigValue').value = config.value;
                document.getElementById('sensitiveConfigDescription').value = config.description || '';
                document.getElementById('sensitiveConfigRotation').value = config.rotation_interval_days || 30;

                // Show modal in edit mode
                this.showSensitiveConfigModal(true);

                showToast('success', 'Config loaded successfully');
            } else {
                showToast('error', response.error || 'Failed to load sensitive config');
            }
        } catch (error) {
            showToast('error', `Error loading sensitive config: ${error.message}`);
        }
    }

    hideSensitiveConfigModal() {
        const modal = document.getElementById('sensitiveConfigModal');
        modal.classList.remove('active');
        // Hide modal after animation completes
        setTimeout(() => modal.style.display = 'none', 300);
    }

    async saveSensitiveConfig() {
        const key = document.getElementById('sensitiveConfigKey').value.trim();
        const value = document.getElementById('sensitiveConfigValue').value.trim();
        const description = document.getElementById('sensitiveConfigDescription').value.trim();
        const rotationDays = parseInt(document.getElementById('sensitiveConfigRotation').value);

        if (!key || !value) {
            showToast('error', 'Key and value are required');
            return;
        }

        try {
            showToast('info', 'Saving sensitive config...');

            const response = await apiPost('/api/settings/sensitive', {
                key: key,
                value: value,
                description: description,
                rotation_days: rotationDays
            });

            if (response.success) {
                showToast('success', 'Sensitive config saved successfully!');
                this.hideSensitiveConfigModal();
                this.loadSensitiveConfigs(); // Reload the list
            } else {
                showToast('error', response.error || 'Failed to save sensitive config');
            }
        } catch (error) {
            showToast('error', `Error saving sensitive config: ${error.message}`);
        }
    }

    async deleteSensitiveConfig(key) {
        if (!confirm(`Are you sure you want to delete the sensitive config "${key}"?`)) {
            return;
        }

        try {
            showToast('info', 'Deleting sensitive config...');

            const response = await fetch(`/api/settings/sensitive/${key}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();

            if (data.success) {
                showToast('success', 'Sensitive config deleted successfully!');
                this.loadSensitiveConfigs(); // Reload the list
            } else {
                showToast('error', data.error || 'Failed to delete sensitive config');
            }
        } catch (error) {
            showToast('error', `Error deleting sensitive config: ${error.message}`);
        }
    }
}
