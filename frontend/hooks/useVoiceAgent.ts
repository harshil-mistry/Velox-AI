import { useState, useRef, useEffect, useCallback } from 'react';

export function useVoiceAgent(serverUrl: string, token: string, ttsProvider: 'deepgram' | 'piper', sttProvider: 'deepgram' | 'gladia', sttLanguage: string) {
    const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [transcript, setTranscript] = useState<{ role: 'user' | 'assistant', content: string }[]>([]);

    // Playback state
    const [sampleRate, setSampleRate] = useState(24000); // Default, updated by server
    const sampleRateRef = useRef(24000);

    const wsRef = useRef<WebSocket | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const workletNodeRef = useRef<AudioWorkletNode | null>(null);
    const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);

    // Audio Queue for playback
    const audioQueueRef = useRef<Float32Array[]>([]);
    const nextStartTimeRef = useRef(0);

    // Keep ttsProvider up to date in the connect callback
    // Ideally, changing ttsProvider should trigger disconnect -> reconnect or just be passed to connect

    const connect = useCallback(async () => {
        setStatus('connecting');

        try {
            // 1. WebSocket
            const ws = new WebSocket(`${serverUrl}?token=${token}&tts_provider=${ttsProvider}&stt_provider=${sttProvider}&stt_language=${sttLanguage}`);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                setStatus('connected');
                console.log(`WS Connected [TTS: ${ttsProvider}]`);
            };

            ws.onmessage = async (event) => {
                const data = event.data;

                if (data instanceof ArrayBuffer) {
                    // Audio Data (PCM16)
                    handleIncomingAudio(data);
                } else {
                    // Text/Config Data (JSON)
                    try {
                        const json = JSON.parse(data);
                        if (json.type === 'transcript') {
                            setTranscript(prev => [...prev, { role: json.role, content: json.content }]);
                        } else if (json.type === 'status') {
                            if (json.content === 'thinking') setIsSpeaking(true);
                            if (json.content === 'listening') setIsSpeaking(false);
                        } else if (json.type === 'config') {
                            // Server tells us the sample rate
                            if (json.sample_rate) {
                                console.log(`Configuring Playback Rate: ${json.sample_rate}`);
                                setSampleRate(json.sample_rate);
                                sampleRateRef.current = json.sample_rate;
                            }
                        }
                    } catch (e) {
                        console.error("JSON Parse error", e);
                    }
                }
            };

            ws.onerror = (e) => {
                console.error("WS Error", e);
                setStatus('error');
            };

            ws.onclose = () => {
                setStatus('idle');
            };

            wsRef.current = ws;

            // 2. Microphone Setup
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            const ctx = new AudioContext({ sampleRate: 16000 });
            await ctx.audioWorklet.addModule('/audio-processor.js');

            const source = ctx.createMediaStreamSource(stream);
            const worklet = new AudioWorkletNode(ctx, 'audio-processor');

            worklet.port.onmessage = (e) => {
                const inputData = e.data as Float32Array;
                const pcm16 = float32ToInt16(inputData);
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(pcm16);
                }
            };

            source.connect(worklet);

            audioContextRef.current = ctx;
            sourceNodeRef.current = source;
            workletNodeRef.current = worklet;

        } catch (e) {
            console.error(e);
            setStatus('error');
        }
    }, [serverUrl, token, ttsProvider, sttProvider, sttLanguage]); // Re-create if provider changes

    const disconnect = useCallback(() => {
        wsRef.current?.close();
        audioContextRef.current?.close();
        setStatus('idle');
    }, []);

    // Playback Logic
    const handleIncomingAudio = (arrayBuffer: ArrayBuffer) => {
        const int16Data = new Int16Array(arrayBuffer);
        const float32Data = new Float32Array(int16Data.length);

        for (let i = 0; i < int16Data.length; i++) {
            float32Data[i] = int16Data[i] / 32768.0;
        }

        audioQueueRef.current.push(float32Data);
        scheduleNextAudioChunk();
    };

    const scheduleNextAudioChunk = () => {
        if (audioQueueRef.current.length === 0 || !audioContextRef.current) return;

        const ctx = audioContextRef.current;
        const chunk = audioQueueRef.current.shift()!;

        // Use state 'sampleRate' OR access it via a ref if state isn't updating fast enough within callback
        // Since schedule functions runs often, using current state might be tricky if it changes mid-stream?
        // Actually, config comes at start. So it should be fine. 
        // NOTE: We need to ensure we use the 'latest' sampleRate.
        // But inside this closure, sampleRate might be stale if not careful.
        // Let's rely on the server validation. Ideally, we should use a Ref for sampleRate.

        // Ref-based access for safety
        // But for React simplicity here, we assume user won't change it mid-call (must disconnect first).

        const audioBuffer = ctx.createBuffer(1, chunk.length, sampleRateRef.current); // Dynamic Rate via Ref
        audioBuffer.getChannelData(0).set(chunk);

        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);

        let startTime = nextStartTimeRef.current;
        if (startTime < ctx.currentTime) startTime = ctx.currentTime;

        source.start(startTime);
        nextStartTimeRef.current = startTime + audioBuffer.duration;
    };

    // Helper
    const float32ToInt16 = (float32: Float32Array) => {
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
            let s = Math.max(-1, Math.min(1, float32[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16;
    };

    return { status, isSpeaking, transcript, connect, disconnect };
}
