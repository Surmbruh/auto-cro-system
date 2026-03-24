class AutoCRO {
    constructor(config) {
        this.apiUrl = config.apiUrl || 'http://localhost:8000';
        this.screenshotTarget = config.screenshotTarget || 'body';
        this.variants = config.variants || []; // CSS selectos for different UI arms
        this.rewardSelectors = config.rewardSelectors || []; // CSS Selectors triggering feedback
        this.sessionId = null;
        this.isLoading = false;
        
        // Hide all variants except the first one (fallback) heavily immediately
        this._hideVariantsBeforeLoad();
    }

    _hideVariantsBeforeLoad() {
        // By default, hide all variants to prevent UI flickering, except the baseline (index 0)
        this.variants.forEach((selector, index) => {
            document.querySelectorAll(selector).forEach(el => {
                if (index !== 0) {
                    el.style.display = 'none';
                }
            });
        });
    }

    async init() {
        console.log("[AutoCRO] Initializing...");
        
        try {
            await this._ensureHtml2CanvasLoaded();
            const blob = await this._takeScreenshot();
            await this._decide(blob);
            this._attachRewardListeners();
        } catch (error) {
            console.error("[AutoCRO] Initialization failed:", error);
            // On failure, baseline (variant 0) remains visible
        }
    }

    async _ensureHtml2CanvasLoaded() {
        if (typeof html2canvas !== 'undefined') return;
        return new Promise((resolve, reject) => {
            console.log("[AutoCRO] Injecting html2canvas via CDN...");
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
            script.onload = resolve;
            script.onerror = () => reject(new Error("Failed to load html2canvas."));
            document.head.appendChild(script);
        });
    }

    async _takeScreenshot() {
        console.log("[AutoCRO] Taking screenshot of", this.screenshotTarget);
        const target = document.querySelector(this.screenshotTarget);
        if (!target) throw new Error("Screenshot target not found");

        const canvas = await html2canvas(target, { useCORS: true, logging: false });
        
        return new Promise((resolve) => {
            // Compress image to JPEG to save transfer size to the VLM model
            canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.8);
        });
    }

    async _decide(imageBlob) {
        this.isLoading = true;
        console.log("[AutoCRO] Sending UI to Backend...");

        const formData = new FormData();
        formData.append('image', imageBlob, 'screenshot.jpeg');

        try {
            const response = await fetch(`${this.apiUrl}/decide`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Invalid /decide response: ${response.status}`);
            }

            const data = await response.json();
            this.sessionId = data.session_id;
            const armIndex = data.arm_index;

            console.log(`[AutoCRO] Bandit chose variant ${armIndex} with confidence ${data.confidence}`);
            this._applyVariant(armIndex);
        } finally {
            this.isLoading = false;
        }
    }

    _applyVariant(armIndex) {
        // Ensure index falls within our variants array bounds
        const safeIndex = (armIndex >= 0 && armIndex < this.variants.length) ? armIndex : 0;
        
        this.variants.forEach((selector, index) => {
            document.querySelectorAll(selector).forEach(el => {
                if (index === safeIndex) {
                    el.style.display = ''; // Restore default display
                    el.classList.add('cro-active-variant');
                } else {
                    el.style.display = 'none';
                    el.classList.remove('cro-active-variant');
                }
            });
        });
    }

    _attachRewardListeners() {
        this.rewardSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                el.addEventListener('click', (e) => {
                    // Send feedback async, don't prevent navigation if it's an anchor tag
                    this.sendFeedback(1.0);
                });
            });
        });
    }

    async sendFeedback(rewardValue) {
        if (!this.sessionId) {
            console.warn("[AutoCRO] Cannot send feedback, no active session_id");
            return;
        }

        console.log(`[AutoCRO] Sending feedback: +${rewardValue} for session ${this.sessionId}`);

        try {
            const response = await fetch(`${this.apiUrl}/feedback`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    reward: rewardValue
                })
            });
            
            if (response.ok) {
                console.log("[AutoCRO] Feedback successfully synchronized.");
                this.sessionId = null; // Clear session so we don't spam feedback
            }
        } catch (error) {
            console.error("[AutoCRO] Error sending feedback:", error);
        }
    }
}

window.AutoCRO = AutoCRO;
