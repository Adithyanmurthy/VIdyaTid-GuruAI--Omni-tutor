/**
 * Offline Status Indicator
 * 
 * Displays offline status and provides verification functionality.
 * Requirements: 4.2, 4.3, 4.4, 4.5
 */

class OfflineIndicator {
    constructor() {
        this.indicator = null;
        this.modal = null;
        this.statusData = null;
        this.updateInterval = null;
        this.init();
    }

    init() {
        this.createIndicator();
        this.createModal();
        this.updateStatus();
        
        // Update status every 30 seconds
        this.updateInterval = setInterval(() => this.updateStatus(), 30000);
    }

    createIndicator() {
        // Create the offline indicator element
        this.indicator = document.createElement('div');
        this.indicator.className = 'offline-indicator offline';
        this.indicator.innerHTML = `
            <div class="status-dot"></div>
            <div class="status-text">
                <span class="status-label">Offline Mode</span>
                <span class="status-sublabel">Active</span>
            </div>
        `;
        
        // Add click handler to show details
        this.indicator.addEventListener('click', () => this.showDetails());
        
        document.body.appendChild(this.indicator);
    }

    createModal() {
        // Create the details modal
        this.modal = document.createElement('div');
        this.modal.className = 'offline-details-modal';
        this.modal.innerHTML = `
            <div class="offline-details-content">
                <div class="offline-details-header">
                    <h2>Offline Status</h2>
                    <button class="offline-details-close">&times;</button>
                </div>
                
                <div class="offline-details-section">
                    <h3>System Status</h3>
                    <div class="offline-status-grid">
                        <div class="offline-status-item">
                            <div class="label">Offline Mode</div>
                            <div class="value" id="offline-mode-status">-</div>
                        </div>
                        <div class="offline-status-item">
                            <div class="label">Internet Available</div>
                            <div class="value" id="internet-status">-</div>
                        </div>
                        <div class="offline-status-item">
                            <div class="label">Monitoring Active</div>
                            <div class="value" id="monitoring-status">-</div>
                        </div>
                        <div class="offline-status-item">
                            <div class="label">External Calls</div>
                            <div class="value" id="external-calls-count">-</div>
                        </div>
                    </div>
                </div>
                
                <div class="offline-details-section">
                    <h3>Verification</h3>
                    <p id="verification-message">Click the button below to verify offline operation.</p>
                    <button class="offline-verify-button" id="verify-offline-btn">
                        Verify Offline Operation
                    </button>
                </div>
                
                <div class="offline-details-section" id="network-calls-section" style="display: none;">
                    <h3>Network Calls Detected</h3>
                    <div class="network-calls-list" id="network-calls-list">
                        <!-- Network calls will be populated here -->
                    </div>
                </div>
            </div>
        `;
        
        // Add event listeners
        const closeBtn = this.modal.querySelector('.offline-details-close');
        closeBtn.addEventListener('click', () => this.hideDetails());
        
        const verifyBtn = this.modal.querySelector('#verify-offline-btn');
        verifyBtn.addEventListener('click', () => this.verifyOffline());
        
        // Close modal when clicking outside
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.hideDetails();
            }
        });
        
        document.body.appendChild(this.modal);
    }

    async updateStatus() {
        try {
            const response = await fetch('/api/offline/status');
            const data = await response.json();
            
            if (data.success) {
                this.statusData = data.status;
                this.updateIndicatorDisplay();
            }
        } catch (error) {
            console.error('Error updating offline status:', error);
            this.setErrorState();
        }
    }

    updateIndicatorDisplay() {
        if (!this.statusData) return;
        
        // Update indicator class based on status
        this.indicator.classList.remove('offline', 'online', 'error');
        
        if (this.statusData.offline_mode_enabled && !this.statusData.external_calls_detected) {
            this.indicator.classList.add('offline');
            this.indicator.querySelector('.status-label').textContent = 'Offline Mode';
            this.indicator.querySelector('.status-sublabel').textContent = 'Active';
        } else if (this.statusData.internet_available) {
            this.indicator.classList.add('online');
            this.indicator.querySelector('.status-label').textContent = 'Internet Detected';
            this.indicator.querySelector('.status-sublabel').textContent = 'Warning';
        } else if (this.statusData.external_calls_detected) {
            this.indicator.classList.add('error');
            this.indicator.querySelector('.status-label').textContent = 'External Calls';
            this.indicator.querySelector('.status-sublabel').textContent = 'Detected';
        }
    }

    setErrorState() {
        this.indicator.classList.remove('offline', 'online');
        this.indicator.classList.add('error');
        this.indicator.querySelector('.status-label').textContent = 'Status Error';
        this.indicator.querySelector('.status-sublabel').textContent = 'Check Console';
    }

    showDetails() {
        this.modal.classList.add('active');
        this.updateModalContent();
    }

    hideDetails() {
        this.modal.classList.remove('active');
    }

    updateModalContent() {
        if (!this.statusData) return;
        
        // Update status values
        const offlineModeStatus = this.modal.querySelector('#offline-mode-status');
        offlineModeStatus.textContent = this.statusData.offline_mode_enabled ? 'Enabled' : 'Disabled';
        offlineModeStatus.className = 'value ' + (this.statusData.offline_mode_enabled ? 'success' : 'warning');
        
        const internetStatus = this.modal.querySelector('#internet-status');
        internetStatus.textContent = this.statusData.internet_available ? 'Yes' : 'No';
        internetStatus.className = 'value ' + (this.statusData.internet_available ? 'warning' : 'success');
        
        const monitoringStatus = this.modal.querySelector('#monitoring-status');
        monitoringStatus.textContent = this.statusData.monitoring_active ? 'Active' : 'Inactive';
        monitoringStatus.className = 'value ' + (this.statusData.monitoring_active ? 'success' : 'warning');
        
        const externalCallsCount = this.modal.querySelector('#external-calls-count');
        externalCallsCount.textContent = this.statusData.external_calls_count || '0';
        externalCallsCount.className = 'value ' + (this.statusData.external_calls_count > 0 ? 'error' : 'success');
    }

    async verifyOffline() {
        const verifyBtn = this.modal.querySelector('#verify-offline-btn');
        const messageEl = this.modal.querySelector('#verification-message');
        
        // Disable button and show loading state
        verifyBtn.disabled = true;
        verifyBtn.textContent = 'Verifying...';
        messageEl.textContent = 'Running offline verification tests...';
        
        try {
            const response = await fetch('/api/offline/verify', {
                method: 'POST'
            });
            const data = await response.json();
            
            if (data.success) {
                const verification = data.verification;
                
                if (verification.status === 'PASS') {
                    messageEl.textContent = '✓ ' + verification.message;
                    messageEl.style.color = 'var(--accent-green)';
                } else {
                    messageEl.textContent = '✗ ' + verification.message;
                    messageEl.style.color = 'var(--accent-red)';
                    
                    // Show external calls if any
                    if (verification.external_calls && verification.external_calls.length > 0) {
                        this.displayNetworkCalls(verification.external_calls);
                    }
                }
            } else {
                messageEl.textContent = 'Error: ' + data.error;
                messageEl.style.color = 'var(--accent-red)';
            }
        } catch (error) {
            console.error('Error verifying offline operation:', error);
            messageEl.textContent = 'Error: ' + error.message;
            messageEl.style.color = 'var(--accent-red)';
        } finally {
            // Re-enable button
            verifyBtn.disabled = false;
            verifyBtn.textContent = 'Verify Offline Operation';
            
            // Update status
            await this.updateStatus();
        }
    }

    displayNetworkCalls(calls) {
        const section = this.modal.querySelector('#network-calls-section');
        const list = this.modal.querySelector('#network-calls-list');
        
        section.style.display = 'block';
        list.innerHTML = '';
        
        calls.forEach(call => {
            const item = document.createElement('div');
            item.className = 'network-call-item' + (call.allowed ? '' : ' external');
            
            const address = call.address || call.url || 'Unknown';
            item.innerHTML = `
                <span class="call-type">${call.type}</span>
                <span class="call-address">${address}</span>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;">
                    ${new Date(call.timestamp).toLocaleString()}
                </div>
            `;
            
            list.appendChild(item);
        });
    }

    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        if (this.indicator) {
            this.indicator.remove();
        }
        if (this.modal) {
            this.modal.remove();
        }
    }
}

// Initialize offline indicator when DOM is ready
let offlineIndicator = null;

document.addEventListener('DOMContentLoaded', () => {
    offlineIndicator = new OfflineIndicator();
});

// Export for use in other scripts
window.OfflineIndicator = OfflineIndicator;
window.offlineIndicator = offlineIndicator;
