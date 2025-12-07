// Progress Dashboard - Track learning progress
document.addEventListener('DOMContentLoaded', () => {
    let currentSubject = 'all';
    let progressData = null;

    // Initialize dashboard
    async function initializeDashboard() {
        await loadProgressData();
        setupEventListeners();
    }

    initializeDashboard();

    // Load progress data from API
    async function loadProgressData() {
        try {
            const response = await fetch('/api/progress', {
                credentials: 'include'
            });

            if (!response.ok) {
                throw new Error('Failed to load progress data');
            }

            progressData = await response.json();
            renderDashboard();
        } catch (error) {
            console.error('Error loading progress:', error);
            showError('Failed to load progress data. Please try again.');
        }
    }

    // Render dashboard
    function renderDashboard() {
        if (!progressData) return;

        renderOverallStats();
        renderSubjectChart(currentSubject);
        renderChapterList(currentSubject);
        renderWeakAreas();
        renderRecommendations();
        renderRecentActivity();
    }

    // Render overall statistics
    function renderOverallStats() {
        const stats = progressData.overall_stats || {};
        
        document.getElementById('total-questions').textContent = stats.total_questions || 0;
        document.getElementById('overall-accuracy').textContent = 
            `${Math.round(stats.overall_accuracy || 0)}%`;
        document.getElementById('topics-mastered').textContent = stats.topics_mastered || 0;
        document.getElementById('study-time').textContent = 
            formatStudyTime(stats.study_time_minutes || 0);
    }

    // Format study time
    function formatStudyTime(minutes) {
        const hours = Math.floor(minutes / 60);
        return `${hours}h`;
    }

    // Render subject chart
    function renderSubjectChart(subject) {
        const chartContainer = document.getElementById('subject-chart');
        chartContainer.innerHTML = '';

        const subjects = subject === 'all' 
            ? ['Physics', 'Chemistry', 'Mathematics', 'Biology']
            : [subject];

        const chartData = subjects.map(subj => {
            const subjectData = progressData.subjects[subj] || {};
            return {
                name: subj,
                accuracy: subjectData.accuracy || 0,
                attempted: subjectData.questions_attempted || 0,
                correct: subjectData.questions_correct || 0
            };
        });

        // Create simple bar chart
        chartData.forEach(data => {
            const barWrapper = document.createElement('div');
            barWrapper.className = 'chart-bar-wrapper';
            
            barWrapper.innerHTML = `
                <div class="chart-label">
                    <span class="subject-name">${data.name}</span>
                    <span class="subject-accuracy">${Math.round(data.accuracy)}%</span>
                </div>
                <div class="chart-bar">
                    <div class="chart-bar-fill" style="width: ${data.accuracy}%"></div>
                </div>
                <div class="chart-stats">
                    <span>${data.correct} / ${data.attempted} correct</span>
                </div>
            `;
            
            chartContainer.appendChild(barWrapper);
        });
    }

    // Render chapter list
    function renderChapterList(subject) {
        const chapterList = document.getElementById('chapter-list');
        chapterList.innerHTML = '';

        const chapters = subject === 'all'
            ? getAllChapters()
            : getChaptersBySubject(subject);

        if (chapters.length === 0) {
            chapterList.innerHTML = '<p class="no-data">No chapter data available</p>';
            return;
        }

        chapters.forEach(chapter => {
            const chapterDiv = document.createElement('div');
            chapterDiv.className = 'chapter-item';
            
            const statusClass = getChapterStatus(chapter.accuracy);
            
            chapterDiv.innerHTML = `
                <div class="chapter-header">
                    <h4>${chapter.subject} - Chapter ${chapter.chapter_number}</h4>
                    <span class="chapter-status ${statusClass}">${statusClass}</span>
                </div>
                <div class="chapter-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${chapter.accuracy}%"></div>
                    </div>
                    <span class="progress-text">${Math.round(chapter.accuracy)}%</span>
                </div>
                <div class="chapter-stats">
                    <span>${chapter.questions_attempted} questions attempted</span>
                    <span>${chapter.questions_correct} correct</span>
                </div>
            `;
            
            chapterList.appendChild(chapterDiv);
        });
    }

    // Get all chapters
    function getAllChapters() {
        const chapters = [];
        Object.keys(progressData.subjects || {}).forEach(subject => {
            const subjectData = progressData.subjects[subject];
            if (subjectData.chapters) {
                Object.keys(subjectData.chapters).forEach(chapterNum => {
                    chapters.push({
                        subject,
                        chapter_number: chapterNum,
                        ...subjectData.chapters[chapterNum]
                    });
                });
            }
        });
        return chapters.sort((a, b) => b.accuracy - a.accuracy);
    }

    // Get chapters by subject
    function getChaptersBySubject(subject) {
        const chapters = [];
        const subjectData = progressData.subjects[subject];
        
        if (subjectData && subjectData.chapters) {
            Object.keys(subjectData.chapters).forEach(chapterNum => {
                chapters.push({
                    subject,
                    chapter_number: chapterNum,
                    ...subjectData.chapters[chapterNum]
                });
            });
        }
        
        return chapters.sort((a, b) => parseInt(a.chapter_number) - parseInt(b.chapter_number));
    }

    // Get chapter status
    function getChapterStatus(accuracy) {
        if (accuracy >= 80) return 'mastered';
        if (accuracy >= 60) return 'good';
        if (accuracy >= 40) return 'needs-work';
        return 'weak';
    }

    // Render weak areas
    function renderWeakAreas() {
        const weakAreasList = document.getElementById('weak-areas-list');
        weakAreasList.innerHTML = '';

        const weakAreas = progressData.weak_areas || [];

        if (weakAreas.length === 0) {
            weakAreasList.innerHTML = '<p class="no-data">Great job! No weak areas identified.</p>';
            return;
        }

        weakAreas.slice(0, 5).forEach(area => {
            const areaDiv = document.createElement('div');
            areaDiv.className = 'weak-area-item';
            
            areaDiv.innerHTML = `
                <div class="area-info">
                    <h4>${area.topic}</h4>
                    <p>${area.subject} - Chapter ${area.chapter}</p>
                </div>
                <div class="area-stats">
                    <span class="accuracy-badge weak">${Math.round(area.accuracy)}%</span>
                    <button class="practice-btn" data-topic="${area.topic}">Practice</button>
                </div>
            `;
            
            weakAreasList.appendChild(areaDiv);
        });

        // Add event listeners to practice buttons
        document.querySelectorAll('.practice-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const topic = e.target.dataset.topic;
                window.location.href = `/question-paper?topic=${encodeURIComponent(topic)}`;
            });
        });
    }

    // Render recommendations
    function renderRecommendations() {
        const recommendationsList = document.getElementById('recommendations-list');
        recommendationsList.innerHTML = '';

        const recommendations = progressData.recommendations || [];

        if (recommendations.length === 0) {
            recommendationsList.innerHTML = '<p class="no-data">Keep up the good work!</p>';
            return;
        }

        recommendations.slice(0, 5).forEach(rec => {
            const recDiv = document.createElement('div');
            recDiv.className = 'recommendation-item';
            
            recDiv.innerHTML = `
                <div class="rec-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                        <polyline points="22 4 12 14.01 9 11.01"></polyline>
                    </svg>
                </div>
                <div class="rec-content">
                    <h4>${rec.topic}</h4>
                    <p>${rec.reason}</p>
                    <button class="study-btn" data-topic="${rec.topic}">Start Studying</button>
                </div>
            `;
            
            recommendationsList.appendChild(recDiv);
        });

        // Add event listeners to study buttons
        document.querySelectorAll('.study-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const topic = e.target.dataset.topic;
                window.location.href = `/?topic=${encodeURIComponent(topic)}`;
            });
        });
    }

    // Render recent activity
    function renderRecentActivity() {
        const activityTimeline = document.getElementById('activity-timeline');
        activityTimeline.innerHTML = '';

        const activities = progressData.recent_activity || [];

        if (activities.length === 0) {
            activityTimeline.innerHTML = '<p class="no-data">No recent activity</p>';
            return;
        }

        activities.slice(0, 10).forEach(activity => {
            const activityDiv = document.createElement('div');
            activityDiv.className = 'activity-item';
            
            const icon = activity.correct ? '✓' : '✗';
            const statusClass = activity.correct ? 'correct' : 'incorrect';
            
            activityDiv.innerHTML = `
                <div class="activity-icon ${statusClass}">${icon}</div>
                <div class="activity-content">
                    <p class="activity-description">${activity.description}</p>
                    <p class="activity-time">${formatTime(activity.timestamp)}</p>
                </div>
            `;
            
            activityTimeline.appendChild(activityDiv);
        });
    }

    // Format timestamp
    function formatTime(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        
        if (diffMins < 60) return `${diffMins} minutes ago`;
        if (diffMins < 1440) return `${Math.floor(diffMins / 60)} hours ago`;
        return `${Math.floor(diffMins / 1440)} days ago`;
    }

    // Setup event listeners
    function setupEventListeners() {
        // Subject tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                currentSubject = e.target.dataset.subject;
                renderSubjectChart(currentSubject);
                renderChapterList(currentSubject);
            });
        });
    }

    // Show error message
    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-banner';
        errorDiv.textContent = message;
        document.querySelector('.progress-container').prepend(errorDiv);
        
        setTimeout(() => errorDiv.remove(), 5000);
    }
});
