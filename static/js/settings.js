// Settings - Configuration and preferences
document.addEventListener('DOMContentLoaded', () => {
    let currentSettings = {};
    let hasUnsavedChanges = false;

    // Initialize
    async function initialize() {
        await loadSettings();
        setupEventListeners();
        updateMemoryUsage();
        loadSystemInfo();
    }

    initialize();

    // Setup event listeners
    function setupEventListeners() {
        // Navigation
        document.querySelectorAll('.settings-nav-item').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const section = e.currentTarget.dataset.section;
                switchSection(section);
            });
        });

        // Save and reset buttons
        document.getElementById('save-settings-btn')?.addEventListener('click', saveSettings);
        document.getElementById('reset-settings-btn')?.addEventListener('click', resetSettings);

        // Track changes
        document.querySelectorAll('input, select').forEach(input => {
            input.addEventListener('change', () => {
                hasUnsavedChanges = true;
                updateSaveButton();
            });
        });

        // Slider value updates
        setupSliders();

        // Special action buttons
        document.getElementById('change-data-location')?.addEventListener('click', changeDataLocation);
        document.getElementById('export-data')?.addEventListener('click', exportData);
        document.getElementById('clear-cache')?.addEventListener('click', clearCache);
        document.getElementById('delete-all-data')?.addEventListener('click', deleteAllData);

        // Warn before leaving with unsaved changes
        window.addEventListener('beforeunload', (e) => {
            if (hasUnsavedChanges) {
                e.preventDefault();
                e.returnValue = '';
            }
        });
    }

    // Setup sliders
    function setupSliders() {
        // Memory limit slider
        const memorySlider = document.getElementById('memory-limit');
        const memoryValue = document.getElementById('memory-limit-value');
        memorySlider?.addEventListener('input', (e) => {
            memoryValue.textContent = `${e.target.value} GB`;
        });

        // Idle timeout slider
        const idleSlider = document.getElementById('idle-timeout');
        const idleValue = document.getElementById('idle-timeout-value');
        idleSlider?.addEventListener('input', (e) => {
            idleValue.textContent = `${e.target.value} min`;
        });

        // Speech speed slider
        const speechSlider = document.getElementById('speech-speed');
        const speechValue = document.getElementById('speech-speed-value');
        speechSlider?.addEventListener('input', (e) => {
            speechValue.textContent = `${parseFloat(e.target.value).toFixed(1)}x`;
        });
    }

    // Switch section
    function switchSection(section) {
        // Update navigation
        document.querySelectorAll('.settings-nav-item').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-section="${section}"]`)?.classList.add('active');

        // Update content
        document.querySelectorAll('.settings-section').forEach(sec => {
            sec.classList.remove('active');
        });
        document.getElementById(`${section}-section`)?.classList.add('active');
    }

    // Load settings
    async function loadSettings() {
        try {
            const response = await fetch('/api/settings', {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to load settings');
            }

            const data = await response.json();
            currentSettings = data.settings || {};
            applySettings(currentSettings);

        } catch (error) {
            console.error('Error loading settings:', error);
            showMessage('Failed to load settings', 'error');
        }
    }

    // Apply settings to UI
    function applySettings(settings) {
        // General settings
        if (settings.target_exam) {
            document.getElementById('target-exam').value = settings.target_exam;
        }
        if (settings.default_language) {
            document.getElementById('default-language').value = settings.default_language;
        }
        if (settings.auto_save !== undefined) {
            document.getElementById('auto-save').checked = settings.auto_save;
        }
        if (settings.show_hints !== undefined) {
            document.getElementById('show-hints').checked = settings.show_hints;
        }

        // Performance settings
        if (settings.memory_limit) {
            document.getElementById('memory-limit').value = settings.memory_limit;
            document.getElementById('memory-limit-value').textContent = `${settings.memory_limit} GB`;
        }
        if (settings.idle_timeout) {
            document.getElementById('idle-timeout').value = settings.idle_timeout;
            document.getElementById('idle-timeout-value').textContent = `${settings.idle_timeout} min`;
        }
        if (settings.response_speed) {
            document.getElementById('response-speed').value = settings.response_speed;
        }
        if (settings.cache_responses !== undefined) {
            document.getElementById('cache-responses').checked = settings.cache_responses;
        }

        // Appearance settings
        if (settings.theme) {
            document.getElementById('theme').value = settings.theme;
            applyTheme(settings.theme);
        }
        if (settings.font_size) {
            document.getElementById('font-size').value = settings.font_size;
        }
        if (settings.compact_mode !== undefined) {
            document.getElementById('compact-mode').checked = settings.compact_mode;
        }
        if (settings.show_animations !== undefined) {
            document.getElementById('show-animations').checked = settings.show_animations;
        }

        // Voice settings
        if (settings.enable_voice_input !== undefined) {
            document.getElementById('enable-voice-input').checked = settings.enable_voice_input;
        }
        if (settings.voice_language) {
            document.getElementById('voice-language').value = settings.voice_language;
        }
        if (settings.enable_tts !== undefined) {
            document.getElementById('enable-tts').checked = settings.enable_tts;
        }
        if (settings.speech_speed) {
            document.getElementById('speech-speed').value = settings.speech_speed;
            document.getElementById('speech-speed-value').textContent = `${parseFloat(settings.speech_speed).toFixed(1)}x`;
        }
        if (settings.voice_type) {
            document.getElementById('voice-type').value = settings.voice_type;
        }

        // Privacy settings
        if (settings.peer_comparison !== undefined) {
            document.getElementById('peer-comparison').checked = settings.peer_comparison;
        }
        if (settings.usage_analytics !== undefined) {
            document.getElementById('usage-analytics').checked = settings.usage_analytics;
        }

        hasUnsavedChanges = false;
        updateSaveButton();
    }

    // Collect settings from UI
    function collectSettings() {
        return {
            // General
            target_exam: document.getElementById('target-exam').value,
            default_language: document.getElementById('default-language').value,
            auto_save: document.getElementById('auto-save').checked,
            show_hints: document.getElementById('show-hints').checked,

            // Performance
            memory_limit: parseInt(document.getElementById('memory-limit').value),
            idle_timeout: parseInt(document.getElementById('idle-timeout').value),
            response_speed: document.getElementById('response-speed').value,
            cache_responses: document.getElementById('cache-responses').checked,

            // Appearance
            theme: document.getElementById('theme').value,
            font_size: document.getElementById('font-size').value,
            compact_mode: document.getElementById('compact-mode').checked,
            show_animations: document.getElementById('show-animations').checked,

            // Voice
            enable_voice_input: document.getElementById('enable-voice-input').checked,
            voice_language: document.getElementById('voice-language').value,
            enable_tts: document.getElementById('enable-tts').checked,
            speech_speed: parseFloat(document.getElementById('speech-speed').value),
            voice_type: document.getElementById('voice-type').value,

            // Privacy
            peer_comparison: document.getElementById('peer-comparison').checked,
            usage_analytics: document.getElementById('usage-analytics').checked
        };
    }

    // Save settings
    async function saveSettings() {
        const settings = collectSettings();

        try {
            const response = await fetch('/api/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ settings }),
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to save settings');
            }

            currentSettings = settings;
            hasUnsavedChanges = false;
            updateSaveButton();
            showMessage('Settings saved successfully', 'success');

            // Apply theme if changed
            applyTheme(settings.theme);

        } catch (error) {
            console.error('Error saving settings:', error);
            showMessage('Failed to save settings', 'error');
        }
    }

    // Reset settings
    async function resetSettings() {
        if (!confirm('Are you sure you want to reset all settings to defaults?')) {
            return;
        }

        try {
            const response = await fetch('/api/settings/reset', {
                method: 'POST',
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to reset settings');
            }

            const data = await response.json();
            currentSettings = data.settings || {};
            applySettings(currentSettings);
            showMessage('Settings reset to defaults', 'success');

        } catch (error) {
            console.error('Error resetting settings:', error);
            showMessage('Failed to reset settings', 'error');
        }
    }

    // Apply theme
    function applyTheme(theme) {
        if (theme === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            theme = prefersDark ? 'dark' : 'light';
        }

        document.body.classList.remove('light-theme', 'dark-theme');
        document.body.classList.add(`${theme}-theme`);
    }

    // Update save button state
    function updateSaveButton() {
        const saveBtn = document.getElementById('save-settings-btn');
        if (saveBtn) {
            saveBtn.disabled = !hasUnsavedChanges;
        }
    }

    // Update memory usage
    async function updateMemoryUsage() {
        try {
            const response = await fetch('/api/system/memory', {
                credentials: 'include'
            });

            if (!response.ok) return;

            const data = await response.json();
            const usagePercent = (data.used / data.total) * 100;

            document.getElementById('memory-usage-text').textContent = 
                `${data.used.toFixed(1)} GB / ${data.total.toFixed(1)} GB`;
            document.getElementById('memory-bar-fill').style.width = `${usagePercent}%`;

        } catch (error) {
            console.error('Error loading memory usage:', error);
        }

        // Update every 5 seconds
        setTimeout(updateMemoryUsage, 5000);
    }

    // Load system info
    async function loadSystemInfo() {
        try {
            const sessionToken = localStorage.getItem('session_token');
            const headers = {
                'Content-Type': 'application/json'
            };
            
            if (sessionToken) {
                headers['Authorization'] = `Bearer ${sessionToken}`;
            }
            
            const response = await fetch('/api/system/info', {
                credentials: 'include',
                headers: headers
            });

            if (!response.ok) {
                console.warn('Failed to load system info:', response.status);
                return;
            }

            const data = await response.json();

            // Update data location
            if (data.data_location) {
                document.getElementById('data-location').textContent = data.data_location;
            }

            // Update diagram count
            if (data.diagram_count) {
                document.getElementById('diagram-count').textContent = 
                    `${data.diagram_count} diagrams`;
            }

        } catch (error) {
            console.error('Error loading system info:', error);
        }
    }

    // Change data location
    async function changeDataLocation() {
        alert('This feature allows you to change where GuruAI stores your data. Please select a new location.');
        
        // In a real implementation, this would open a directory picker
        // For now, just show a message
        showMessage('Data location change is not yet implemented', 'info');
    }

    // Export data
    async function exportData() {
        try {
            const response = await fetch('/api/data/export', {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to export data');
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `guruai_data_${new Date().toISOString().split('T')[0]}.zip`;
            a.click();
            URL.revokeObjectURL(url);

            showMessage('Data exported successfully', 'success');

        } catch (error) {
            console.error('Error exporting data:', error);
            showMessage('Failed to export data', 'error');
        }
    }

    // Clear cache
    async function clearCache() {
        if (!confirm('Are you sure you want to clear the cache? This will free up space but may slow down initial responses.')) {
            return;
        }

        try {
            const response = await fetch('/api/cache/clear', {
                method: 'POST',
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to clear cache');
            }

            const data = await response.json();
            showMessage(`Cache cleared. Freed ${data.freed_space} MB`, 'success');

        } catch (error) {
            console.error('Error clearing cache:', error);
            showMessage('Failed to clear cache', 'error');
        }
    }

    // Delete all data
    async function deleteAllData() {
        const confirmation = prompt('This will permanently delete ALL your data including progress, settings, and history. Type "DELETE" to confirm:');
        
        if (confirmation !== 'DELETE') {
            return;
        }

        try {
            const response = await fetch('/api/data/delete-all', {
                method: 'POST',
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to delete data');
            }

            showMessage('All data deleted. Redirecting to login...', 'success');
            
            setTimeout(() => {
                window.location.href = '/auth';
            }, 2000);

        } catch (error) {
            console.error('Error deleting data:', error);
            showMessage('Failed to delete data', 'error');
        }
    }

    // Show message
    function showMessage(message, type) {
        const messageDiv = document.getElementById('settings-message');
        if (!messageDiv) return;

        messageDiv.textContent = message;
        messageDiv.className = `settings-message ${type}`;
        messageDiv.classList.remove('hidden');

        setTimeout(() => {
            messageDiv.classList.add('hidden');
        }, 5000);
    }

    // Confirmation modal
    function showConfirmModal(title, message, onConfirm) {
        const modal = document.getElementById('confirm-modal');
        if (!modal) return;

        document.getElementById('confirm-title').textContent = title;
        document.getElementById('confirm-message').textContent = message;

        modal.classList.remove('hidden');

        const yesBtn = document.getElementById('confirm-yes');
        const noBtn = document.getElementById('confirm-no');

        const handleYes = () => {
            onConfirm();
            modal.classList.add('hidden');
            cleanup();
        };

        const handleNo = () => {
            modal.classList.add('hidden');
            cleanup();
        };

        const cleanup = () => {
            yesBtn.removeEventListener('click', handleYes);
            noBtn.removeEventListener('click', handleNo);
        };

        yesBtn.addEventListener('click', handleYes);
        noBtn.addEventListener('click', handleNo);
    }
});
