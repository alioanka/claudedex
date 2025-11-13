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
        document.querySelectorAll('.category-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e));
        });

        const saveButton = document.getElementById('saveSettings');
        if (saveButton) {
            saveButton.addEventListener('click', () => this.saveAllSettings());
        }
    }

    switchTab(event) {
        event.preventDefault();
        const clickedTab = event.currentTarget;
        const category = clickedTab.dataset.category;

        // Deactivate all tabs and hide all sections
        document.querySelectorAll('.category-tab').forEach(tab => tab.classList.remove('active'));
        document.querySelectorAll('.settings-section').forEach(section => {
            section.style.display = 'none';
        });

        // Activate the clicked tab and show the corresponding section
        clickedTab.classList.add('active');
        const sectionId = `${category}-settings-section`;
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = 'block';
        }
    }

    async loadSettings() {
        try {
            const response = await fetch('/api/settings/all');
            if (!response.ok) {
                throw new Error(`Failed to fetch settings: ${response.statusText}`);
            }
            const result = await response.json();

            if (result.success && result.data) {
                this.settings = result.data;
                this.initialSettings = JSON.parse(JSON.stringify(result.data)); // Deep copy
                this.populateForm(this.settings);
                console.log('Settings loaded successfully.');
            } else {
                console.error('Failed to load settings:', result.error);
            }
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    populateForm(data) {
        document.querySelectorAll('.settings-input').forEach(input => {
            const category = input.closest('[data-category]').dataset.category;
            const key = input.id;
            if (data[category] && data[category].hasOwnProperty(key)) {
                if (input.type === 'checkbox') {
                    input.checked = data[category][key];
                } else {
                    input.value = data[category][key];
                }
            }
        });
    }

    async saveAllSettings() {
        const sections = document.querySelectorAll('.settings-section .card-body');
        let allUpdatesSucceeded = true;

        for (const section of sections) {
            const category = section.dataset.category;
            if (!category) continue;

            const updates = {};
            const inputs = section.querySelectorAll('.settings-input');

            inputs.forEach(input => {
                const key = input.id;
                let value;
                if (input.type === 'checkbox') {
                    value = input.checked;
                } else if (input.type === 'number') {
                    value = parseFloat(input.value);
                } else {
                    value = input.value;
                }
                updates[key] = value;
            });

            if (Object.keys(updates).length > 0) {
                const success = await this.sendUpdateRequest(category, updates);
                if (!success) {
                    allUpdatesSucceeded = false;
                }
            }
        }

        if (allUpdatesSucceeded) {
            showToast('success', 'All settings saved successfully!');
            // Optionally, reload settings to confirm they've been applied
            this.loadSettings();
        } else {
            showToast('error', 'One or more settings categories failed to save.');
        }
    }

    async sendUpdateRequest(category, updates) {
        try {
            const response = await fetch('/api/settings/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    config_type: category,
                    updates: updates,
                }),
            });

            if (!response.ok) {
                throw new Error(`API returned status ${response.status}`);
            }

            const result = await response.json();

            if (result.success) {
                console.log(`Settings for '${category}' updated successfully.`);
                return true;
            } else {
                showToast('error', `Failed to update settings for '${category}': ${result.error}`);
                return false;
            }
        } catch (error) {
            showToast('error', `Error saving settings for '${category}': ${error.message}`);
            return false;
        }
    }
}
