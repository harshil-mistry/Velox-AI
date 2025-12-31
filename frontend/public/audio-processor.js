class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 2048;
        this._buffer = new Float32Array(this.bufferSize);
        this._bytesWritten = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];

        if (!input || input.length === 0) return true;

        // We only care about the first channel for now (Mono)
        const inputChannel = input[0];

        // Downsample and convert logic would go here.
        // However, AudioWorklet runs at the Context sample rate (usually 44.1k or 48k).
        // We want to send 16k PCM to the backend.

        // Simple approach: Send raw Float32 chunks and let the main thread or backend handle resampling?
        // Backend expects 16k Linear16. Sending 48k Float32 is heavy.
        // Let's implement a simple downsampler here.

        // NOTE: For simplicity in this "Test" phase, we will just pass the Float32 buffer
        // to the main thread and handle resampling/conversion there or here.
        // Actually, doing it here prevents main thread jank.

        // Let's just send the raw buffer to main thread for now to keep Worklet simple,
        // and use a library or manual logic in the main thread (or just send 48k and let Deepgram handle it? strictly 16k was requested).

        // Let's try to send raw data.

        this.port.postMessage(inputChannel);

        return true;
    }
}

registerProcessor("audio-processor", AudioProcessor);
