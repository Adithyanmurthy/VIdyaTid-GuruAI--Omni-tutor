/**
 * Video Generator JavaScript
 * Handles UI interactions for 2D Animated Explainer Video generation
 */

class VideoGenerator {
    constructor() {
        this.topicInput = document.getElementById('topic-input');
        this.subjectSelect = document.getElementById('subject-select');
        this.durationInput = document.getElementById('duration-input');
        this.previewBtn = document.getElementById('preview-btn');
        this.generateBtn = document.getElementById('generate-btn');
        this.statusSection = document.getElementById('status-section');
        this.scriptPreview = document.getElementById('script-preview');
        this.resultSection = document.getElementById('result-section');
        this.topicsGrid = document.getElementById('topics-grid');
        this.videosList = document.getElementById('videos-list');
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadQuickTopics();
        this.loadVideosList();
        this.checkStatus();
    }
    
    bindEvents() {
        this.previewBtn.addEventListener('click', () => this.previewScript());
        this.generateBtn.addEventListener('click', () => this.generateVideo());
        
        document.getElementById('close-preview')?.addEventListener('click', () => {
            this.scriptPreview.classList.add('hidden');
        });
        
        document.getElementById('new-video-btn')?.addEventListener('click', () => {
            this.resultSection.classList.add('hidden');
            this.topicInput.value = '';
            this.topicInput.focus();
        });
        
        // Enter key to generate
        this.topicInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.generateVideo();
        });
    }
    
    async checkStatus() {
        try {
            const response = await fetch('/video/api/status');
            const data = await response.json();
            console.log('Video Generator Status:', data);
        } catch (error) {
            console.error('Status check failed:', error);
        }
    }
    
    async loadQuickTopics() {
        try {
            const response = await fetch('/video/api/topics');
            const data = await response.json();
            
            if (data.success && data.topics) {
                this.renderQuickTopics(data.topics);
            }
        } catch (error) {
            console.error('Failed to load topics:', error);
            this.renderDefaultTopics();
        }
    }
    
    renderQuickTopics(topics) {
        const icons = {
            physics: '⚛️',
            chemistry: '⚗️',
            biology: '🧬',
            maths: '📐'
        };
        
        this.topicsGrid.innerHTML = topics.slice(0, 8).map(item => `
            <div class="topic-chip" data-topic="${item.topic}" data-subject="${item.subject}">
                <span class="chip-icon">${icons[item.subject] || '📚'}</span>
                ${item.topic}
            </div>
        `).join('');
        
        // Bind click events
        this.topicsGrid.querySelectorAll('.topic-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                this.topicInput.value = chip.dataset.topic;
                this.subjectSelect.value = chip.dataset.subject.charAt(0).toUpperCase() + chip.dataset.subject.slice(1);
            });
        });
    }
    
    renderDefaultTopics() {
        const defaultTopics = [
            { topic: 'DNA Replication', subject: 'biology' },
            { topic: "Newton's Laws", subject: 'physics' },
            { topic: 'Chemical Bonding', subject: 'chemistry' },
            { topic: 'Quadratic Equations', subject: 'maths' }
        ];
        this.renderQuickTopics(defaultTopics);
    }

    
    async previewScript() {
        const topic = this.topicInput.value.trim();
        if (!topic) {
            this.showNotification('Please enter a topic', 'error');
            this.topicInput.focus();
            return;
        }
        
        this.previewBtn.classList.add('btn-loading');
        this.previewBtn.disabled = true;
        
        try {
            const response = await fetch('/video/api/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    topic: topic,
                    subject: this.subjectSelect.value || null
                })
            });
            
            const data = await response.json();
            
            if (data.success && data.script) {
                this.showScriptPreview(data.script);
            } else {
                this.showNotification(data.error || 'Failed to generate preview', 'error');
            }
        } catch (error) {
            console.error('Preview error:', error);
            this.showNotification('Failed to connect to server', 'error');
        } finally {
            this.previewBtn.classList.remove('btn-loading');
            this.previewBtn.disabled = false;
        }
    }
    
    showScriptPreview(script) {
        const content = document.getElementById('preview-content');
        
        let html = `
            <div class="preview-meta">
                <p><strong>Topic:</strong> ${script.topic}</p>
                <p><strong>Subject:</strong> ${script.subject}</p>
                <p><strong>Duration:</strong> ${script.duration_info?.formatted || 'N/A'}</p>
            </div>
            <hr style="border-color: rgba(59, 130, 246, 0.3); margin: 1rem 0;">
        `;
        
        if (script.scenes && script.scenes.length > 0) {
            script.scenes.forEach((scene, index) => {
                html += `
                    <div class="scene-card">
                        <div class="scene-header">
                            <span class="scene-number">Scene ${index + 1}</span>
                            <span class="scene-duration">${scene.duration}s</span>
                        </div>
                        <p class="scene-narration">${scene.narration}</p>
                    </div>
                `;
            });
        }
        
        content.innerHTML = html;
        this.scriptPreview.classList.remove('hidden');
    }
    
    async generateVideo() {
        const topic = this.topicInput.value.trim();
        if (!topic) {
            this.showNotification('Please enter a topic', 'error');
            this.topicInput.focus();
            return;
        }
        
        this.generateBtn.classList.add('btn-loading');
        this.generateBtn.disabled = true;
        this.showStatus('Generating video...', 'This may take a minute');
        
        try {
            const response = await fetch('/video/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    topic: topic,
                    subject: this.subjectSelect.value || null,
                    duration: this.durationInput.value ? parseInt(this.durationInput.value) : null
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showResult(data);
                this.loadVideosList();
            } else {
                this.showNotification(data.error || 'Failed to generate video', 'error');
            }
        } catch (error) {
            console.error('Generation error:', error);
            this.showNotification('Failed to connect to server', 'error');
        } finally {
            this.generateBtn.classList.remove('btn-loading');
            this.generateBtn.disabled = false;
            this.hideStatus();
        }
    }
    
    showStatus(title, message) {
        document.getElementById('status-title').textContent = title;
        document.getElementById('status-message').textContent = message;
        document.getElementById('progress-fill').style.width = '60%';
        this.statusSection.classList.remove('hidden');
    }
    
    hideStatus() {
        this.statusSection.classList.add('hidden');
    }
    
    showResult(data) {
        document.getElementById('result-topic').textContent = data.topic;
        document.getElementById('result-duration').textContent = `${data.duration_seconds}s`;
        document.getElementById('result-scenes').textContent = data.script?.total_scenes || 'N/A';
        
        const downloadLink = document.getElementById('download-link');
        if (data.video_path) {
            const filename = data.video_path.split('/').pop();
            downloadLink.href = `/video/api/download/${filename}`;
        }
        
        this.resultSection.classList.remove('hidden');
    }
    
    async loadVideosList() {
        try {
            const response = await fetch('/video/api/videos');
            const data = await response.json();
            
            if (data.success && data.videos.length > 0) {
                this.renderVideosList(data.videos);
            } else {
                this.videosList.innerHTML = '<p class="empty-message">No videos generated yet</p>';
            }
        } catch (error) {
            console.error('Failed to load videos:', error);
        }
    }
    
    renderVideosList(videos) {
        this.videosList.innerHTML = videos.map(video => `
            <div class="video-item">
                <div class="video-item-info">
                    <span class="video-item-icon">🎬</span>
                    <div class="video-item-details">
                        <h4>${video.filename}</h4>
                        <span class="video-item-meta">${video.size_mb} MB • ${new Date(video.created).toLocaleDateString()}</span>
                    </div>
                </div>
                <a href="/video/api/download/${video.filename}" class="btn-small" download>Download</a>
            </div>
        `).join('');
    }
    
    showNotification(message, type = 'info') {
        // Simple alert for now - could be enhanced with toast notifications
        alert(message);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.videoGenerator = new VideoGenerator();
});
