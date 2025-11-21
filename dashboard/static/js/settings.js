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
            if (category === 'account') {
                pageActions.style.display = 'none';
            } else {
                pageActions.style.display = 'flex';
            }
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
            const value = categoryData[key];
            const settingElement = this.createSettingElement(category, key, value);
            if (settingElement) {
                fragment.appendChild(settingElement);
            }
        }
        return fragment;
    }

    createSettingElement(category, key, value) {
        const formGroup = document.createElement('div');
        formGroup.className = 'form-group';

        const label = document.createElement('label');
        label.className = 'form-label';
        label.setAttribute('for', `${category}-${key}`);
        label.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

        let input;
        if (typeof value === 'boolean') {
            input = document.createElement('input');
            input.type = 'checkbox';
            input.checked = value;
            input.className = 'form-checkbox';
        } else if (typeof value === 'number') {
            input = document.createElement('input');
            input.type = 'number';
            input.value = value;
            input.className = 'form-control';
        } else if (typeof value === 'string') {
            input = document.createElement('input');
            input.type = 'text';
            input.value = value;
            input.className = 'form-control';
        } else {
            // For nested objects or arrays, display as readonly JSON
            input = document.createElement('textarea');
            input.value = JSON.stringify(value, null, 2);
            input.className = 'form-control';
            input.rows = 4;
            input.readOnly = true;
        }

        input.id = `${category}-${key}`;
        input.dataset.category = category;
        input.dataset.key = key;

        formGroup.appendChild(label);
        formGroup.appendChild(input);

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
                let value;
                if (input.type === 'checkbox') {
                    value = input.checked;
                } else if (input.type === 'number') {
                    value = parseFloat(input.value);
                } else if (input.tagName === 'TEXTAREA') {
                    // Skip read-only textareas
                    return;
                } else {
                    value = input.value;
                }

                // Only include changed values
                if (this.initialSettings[category][key] !== value) {
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
}
