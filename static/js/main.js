/**
 * VoxGuard — Federated AI Voice Deepfake Detection
 * Dashboard Interactivity, Live Microphone Recorder & Web Audio Visualizer
 */

let selectedFile = null;
let selectedPreset = null;
let currentTab = 'real';
let inputMode = 'mic';

// Microphone Recording Variables
let mediaRecorder = null;
let audioChunks = [];
let recTimerInterval = null;
let recSeconds = 0;
let recStream = null;

document.addEventListener('DOMContentLoaded', () => {
    initDropzone();
    initMicRecorder();
    loadPresetSamples();
    loadFLMetrics();
});

/* ----------------------------------------------------
   1. Input Mode Switching (Mic vs File Upload)
---------------------------------------------------- */
function switchInputMode(mode) {
    inputMode = mode;
    const micBtn = document.getElementById('mode-mic-btn');
    const fileBtn = document.getElementById('mode-file-btn');
    const micBox = document.getElementById('mic-recorder-box');
    const dropzone = document.getElementById('dropzone');

    micBtn.classList.toggle('active', mode === 'mic');
    fileBtn.classList.toggle('active', mode === 'file');

    micBox.classList.toggle('hidden', mode !== 'mic');
    dropzone.classList.toggle('hidden', mode !== 'file');
}

/* ----------------------------------------------------
   2. Live Microphone Recorder & WebAudio WAV Encoder
---------------------------------------------------- */
function initMicRecorder() {
    const startBtn = document.getElementById('start-rec-btn');
    const stopBtn = document.getElementById('stop-rec-btn');

    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
}

async function startRecording() {
    try {
        recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(recStream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const rawBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            const wavBlob = await convertBlobToWav(rawBlob);

            selectedFile = new File([wavBlob], 'live_voice_recording.wav', { type: 'audio/wav' });
            selectedPreset = null;

            document.querySelectorAll('.preset-item').forEach(el => el.classList.remove('selected'));

            const playerWrapper = document.getElementById('player-wrapper');
            const audioPlayer = document.getElementById('audio-player');
            const filenameLabel = document.getElementById('current-filename');
            const formatBadge = document.getElementById('audio-format-badge');
            const analyzeBtn = document.getElementById('analyze-btn');

            filenameLabel.innerHTML = `<i class="fa-solid fa-microphone"></i> live_voice_recording.wav`;
            formatBadge.textContent = 'MIC WAV';

            const objectUrl = URL.createObjectURL(selectedFile);
            audioPlayer.src = objectUrl;
            playerWrapper.classList.remove('hidden');
            analyzeBtn.disabled = false;

            setupCanvasWaveform(audioPlayer);
        };

        mediaRecorder.start();

        // UI Recording State
        document.getElementById('start-rec-btn').classList.add('hidden');
        document.getElementById('stop-rec-btn').classList.remove('hidden');
        document.getElementById('mic-pulse-ring').classList.add('is-recording');
        document.getElementById('rec-status-text').textContent = 'Recording live voice... speak now';
        
        const recTimer = document.getElementById('rec-timer');
        recTimer.classList.remove('hidden');
        recSeconds = 0;
        recTimer.textContent = 'REC 00:00';

        recTimerInterval = setInterval(() => {
            recSeconds++;
            const mins = String(Math.floor(recSeconds / 60)).padStart(2, '0');
            const secs = String(recSeconds % 60).padStart(2, '0');
            recTimer.textContent = `REC ${mins}:${secs}`;

            if (recSeconds >= 30) {
                stopRecording();
            }
        }, 1000);

    } catch (err) {
        console.error('Microphone access denied:', err);
        alert('Microphone access is required for live voice input. Please grant microphone permission in your browser.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    if (recStream) {
        recStream.getTracks().forEach(track => track.stop());
    }

    clearInterval(recTimerInterval);

    document.getElementById('start-rec-btn').classList.remove('hidden');
    document.getElementById('stop-rec-btn').classList.add('hidden');
    document.getElementById('mic-pulse-ring').classList.remove('is-recording');
    document.getElementById('rec-status-text').textContent = 'Voice recording captured! Ready for detection.';
    document.getElementById('rec-timer').classList.add('hidden');
}

/**
 * Pure JS Blob to 16-bit PCM WAV Converter via AudioContext
 */
async function convertBlobToWav(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decodedData = await audioCtx.decodeAudioData(arrayBuffer);

    return audioBufferToWavBlob(decodedData);
}

function audioBufferToWavBlob(buffer) {
    const numChannels = 1; // mono for speech
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    const samples = buffer.getChannelData(0);
    const bufferLength = samples.length;
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    
    const wavBuffer = new ArrayBuffer(44 + bufferLength * bytesPerSample);
    const view = new DataView(wavBuffer);
    
    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + bufferLength * bytesPerSample, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, 'data');
    view.setUint32(40, bufferLength * bytesPerSample, true);
    
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    
    return new Blob([view], { type: 'audio/wav' });
}

/* ----------------------------------------------------
   3. Dropzone & File Selection Setup
---------------------------------------------------- */
function initDropzone() {
    const dropzone = document.getElementById('dropzone');
    const audioInput = document.getElementById('audio-input');
    const browseBtn = document.getElementById('browse-btn');

    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        audioInput.click();
    });

    dropzone.addEventListener('click', () => audioInput.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleSelectedFile(e.dataTransfer.files[0]);
        }
    });

    audioInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    document.getElementById('analyze-btn').addEventListener('click', runDetection);
}

function handleSelectedFile(file) {
    selectedFile = file;
    selectedPreset = null;

    document.querySelectorAll('.preset-item').forEach(el => el.classList.remove('selected'));

    const playerWrapper = document.getElementById('player-wrapper');
    const audioPlayer = document.getElementById('audio-player');
    const filenameLabel = document.getElementById('current-filename');
    const formatBadge = document.getElementById('audio-format-badge');
    const analyzeBtn = document.getElementById('analyze-btn');

    filenameLabel.innerHTML = `<i class="fa-solid fa-file-audio"></i> ${file.name}`;
    formatBadge.textContent = file.name.split('.').pop().toUpperCase();

    const objectUrl = URL.createObjectURL(file);
    audioPlayer.src = objectUrl;
    playerWrapper.classList.remove('hidden');
    analyzeBtn.disabled = false;

    setupCanvasWaveform(audioPlayer);
}

/* ----------------------------------------------------
   4. Preset Dataset Selector
---------------------------------------------------- */
async function loadPresetSamples() {
    try {
        const res = await fetch('/api/presets');
        const data = await res.json();
        
        window.presetData = data;
        document.getElementById('real-count').textContent = data.real_samples.length;
        document.getElementById('fake-count').textContent = data.fake_samples.length;

        renderPresetList();
    } catch (err) {
        console.error('Failed to load presets:', err);
    }
}

function switchPresetTab(tab) {
    currentTab = tab;
    document.getElementById('tab-real-btn').classList.toggle('active', tab === 'real');
    document.getElementById('tab-fake-btn').classList.toggle('active', tab === 'fake');
    renderPresetList();
}

function renderPresetList() {
    const listContainer = document.getElementById('preset-list');
    listContainer.innerHTML = '';

    const files = currentTab === 'real' 
        ? (window.presetData?.real_samples || [])
        : (window.presetData?.fake_samples || []);

    if (files.length === 0) {
        listContainer.innerHTML = '<div class="loading-spinner">No samples available in directory</div>';
        return;
    }

    files.forEach(filename => {
        const item = document.createElement('div');
        item.className = 'preset-item';
        item.innerHTML = `
            <span class="name"><i class="fa-solid fa-play"></i> ${filename}</span>
            <span class="badge badge-small">${currentTab === 'real' ? 'REAL' : 'FAKE'}</span>
        `;
        item.onclick = () => selectPreset(filename, currentTab, item);
        listContainer.appendChild(item);
    });
}

function selectPreset(filename, category, element) {
    selectedPreset = { filename, category };
    selectedFile = null;

    document.querySelectorAll('.preset-item').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');

    const playerWrapper = document.getElementById('player-wrapper');
    const audioPlayer = document.getElementById('audio-player');
    const filenameLabel = document.getElementById('current-filename');
    const formatBadge = document.getElementById('audio-format-badge');
    const analyzeBtn = document.getElementById('analyze-btn');

    filenameLabel.innerHTML = `<i class="fa-solid fa-music"></i> ${filename}`;
    formatBadge.textContent = filename.split('.').pop().toUpperCase();

    audioPlayer.src = `/api/sample/${category}/${filename}`;
    playerWrapper.classList.remove('hidden');
    analyzeBtn.disabled = false;

    setupCanvasWaveform(audioPlayer);
}

/* ----------------------------------------------------
   5. Run Deepfake Detection (API Call)
---------------------------------------------------- */
async function runDetection() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultBody = document.getElementById('result-body');

    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Processing Federated Model...';

    try {
        let response;
        if (selectedFile) {
            const formData = new FormData();
            formData.append('audio', selectedFile);
            response = await fetch('/api/predict_upload', {
                method: 'POST',
                body: formData
            });
        } else if (selectedPreset) {
            response = await fetch('/api/predict_preset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(selectedPreset)
            });
        }

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        renderDetectionResult(data);

        resultPlaceholder.classList.add('hidden');
        resultBody.classList.remove('hidden');
    } catch (err) {
        console.error('Detection error:', err);
        alert('Failed to connect to VoxGuard backend server.');
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Federated Deepfake Detection';
    }
}

function renderDetectionResult(data) {
    const banner = document.getElementById('verdict-banner');
    const icon = document.getElementById('verdict-icon');
    const title = document.getElementById('verdict-title');
    const desc = document.getElementById('verdict-desc');

    if (data.is_fake) {
        banner.className = 'verdict-banner verdict-fake';
        icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
        title.textContent = '🚨 AI-GENERATED DEEPFAKE';
        desc.textContent = `Federated CNN-LSTM Model indicates high probability of synthetic audio generation (${data.confidence}% confidence).`;
    } else {
        banner.className = 'verdict-banner verdict-real';
        icon.innerHTML = '<i class="fa-solid fa-shield-check"></i>';
        title.textContent = '✅ GENUINE HUMAN VOICE';
        desc.textContent = `Federated CNN-LSTM Model verified natural acoustic variations (${data.confidence}% confidence).`;
    }

    // Gauge Score
    document.getElementById('gauge-score').textContent = `${data.prob_fake}%`;
    const dashoffset = 126 - (126 * (data.prob_fake / 100));
    document.getElementById('gauge-fill-path').style.strokeDashoffset = dashoffset;

    // Progress Bars
    document.getElementById('prob-real-bar').style.width = `${data.prob_real}%`;
    document.getElementById('prob-real-val').textContent = `${data.prob_real}%`;

    document.getElementById('prob-fake-bar').style.width = `${data.prob_fake}%`;
    document.getElementById('prob-fake-val').textContent = `${data.prob_fake}%`;

    // Spectrogram Image
    document.getElementById('spectrogram-img').src = `data:image/png;base64,${data.spectrogram}`;

    // Metrics
    const stats = data.acoustic_stats || {};
    document.getElementById('metric-duration').textContent = stats.duration ? `${stats.duration}s` : '-';
    document.getElementById('metric-frames').textContent = stats.mfcc_timesteps || '-';
    document.getElementById('metric-energy').textContent = stats.rms_energy || '-';
    document.getElementById('metric-centroid').textContent = stats.spectral_centroid ? `${stats.spectral_centroid} Hz` : '-';
}

/* ----------------------------------------------------
   6. Canvas Audio Waveform Visualizer
---------------------------------------------------- */
function setupCanvasWaveform(audioEl) {
    const canvas = document.getElementById('waveform-canvas');
    const ctx = canvas.getContext('2d');
    drawWaveform(ctx, canvas.width, canvas.height);
}

function drawWaveform(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#00f2fe';
    const barWidth = 3;
    const gap = 2;
    const numBars = Math.floor(width / (barWidth + gap));

    for (let i = 0; i < numBars; i++) {
        const barHeight = Math.sin(i * 0.15) * 18 + Math.cos(i * 0.3) * 10 + 25;
        const x = i * (barWidth + gap);
        const y = (height - barHeight) / 2;

        ctx.fillStyle = i % 3 === 0 ? '#00f2fe' : '#7928ca';
        ctx.fillRect(x, y, barWidth, barHeight);
    }
}

/* ----------------------------------------------------
   7. Federated Learning Edge Clients & Simulation
---------------------------------------------------- */
async function loadFLMetrics() {
    try {
        const res = await fetch('/api/fl_metrics');
        const data = await res.json();
        renderClientNodes(data.clients || []);
    } catch (err) {
        console.error('Failed to load FL metrics:', err);
    }
}

function renderClientNodes(clients) {
    const container = document.getElementById('client-nodes-grid');
    container.innerHTML = '';

    const icons = [
        'fa-mobile-screen',
        'fa-volume-high',
        'fa-phone-volume',
        'fa-desktop',
        'fa-microchip'
    ];

    clients.forEach((client, idx) => {
        const card = document.createElement('div');
        card.className = 'client-node-card';
        card.innerHTML = `
            <div class="icon"><i class="fa-solid ${icons[idx % icons.length]}"></i></div>
            <h5>${client.name}</h5>
            <p>${client.samples} Samples</p>
            <p style="color: var(--primary); margin-top:2px;">${client.privacy}</p>
        `;
        container.appendChild(card);
    });
}

async function simulateFLRound() {
    try {
        const res = await fetch('/api/simulate_fl_round', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ round: 10 })
        });
        const data = await res.json();

        alert(`🔄 FedAvg Communication Round Completed Successfully!\nGlobal Accuracy: ${data.global_accuracy}%\nGlobal Loss: ${data.global_loss}\nWeights Aggregated Across 5 Edge Devices without Raw Audio Data Transfer!`);
    } catch (err) {
        console.error('FL simulation failed:', err);
    }
}
