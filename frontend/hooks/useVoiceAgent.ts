import { useState, useRef, useEffect, useCallback } from 'react';

const SAMPLE_RATE = 16000;

export function useVoiceAgent(serverUrl: string, token: string) {
    const [status, setStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [transcript, setTranscript] = useState<{ role: 'user' | 'assistant', content: string }[]>([]);

    const wsRef = useRef<WebSocket | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const workletNodeRef = useRef<AudioWorkletNode | null>(null);
    const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);

    // Audio Queue for playback
    const audioQueueRef = useRef<Float32Array[]>([]);
    const isPlayingRef = useRef(false);
    const nextStartTimeRef = useRef(0);

    const connect = useCallback(async () => {
        setStatus('connecting');

        try {
            // 1. WebSocket
            const ws = new WebSocket(`${serverUrl}?token=${token}`);
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {
                setStatus('connected');
                console.log('WS Connected');
            };

            ws.onmessage = async (event) => {
                const data = event.data;

                if (data instanceof ArrayBuffer) {
                    // Audio Data (PCM16)
                    handleIncomingAudio(data);
                } else {
                    // Text Data (JSON)
                    try {
                        const json = JSON.parse(data);
                        if (json.type === 'transcript') {
                            setTranscript(prev => [...prev, { role: json.role, content: json.content }]);
                        } else if (json.type === 'status') {
                            if (json.content === 'thinking') setIsSpeaking(true);
                            if (json.content === 'listening') setIsSpeaking(false);
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
                    sampleRate: 16000, // Try to request 16k
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            const ctx = new AudioContext({ sampleRate: 16000 }); // Force 16k context if supported
            await ctx.audioWorklet.addModule('/audio-processor.js');

            const source = ctx.createMediaStreamSource(stream);
            const worklet = new AudioWorkletNode(ctx, 'audio-processor');

            worklet.port.onmessage = (e) => {
                // Incoming raw Float32 (usually 44100/48000 depending on Hardware, checking ctx.sampleRate)
                const inputData = e.data as Float32Array;

                // Convert Float32 -> Int16
                const pcm16 = float32ToInt16(inputData);

                // Send to WS
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(pcm16);
                }
            };

            source.connect(worklet);
            // worklet.connect(ctx.destination); // Don't connect mic to speakers! (Feedback)

            audioContextRef.current = ctx;
            sourceNodeRef.current = source;
            workletNodeRef.current = worklet;

        } catch (e) {
            console.error(e);
            setStatus('error');
        }
    }, [serverUrl, token]);

    const disconnect = useCallback(() => {
        wsRef.current?.close();
        audioContextRef.current?.close();
        setStatus('idle');
    }, []);

    // Playback Logic
    const handleIncomingAudio = (arrayBuffer: ArrayBuffer) => {
        // Raw Linear16 PCM
        const int16Data = new Int16Array(arrayBuffer);
        const float32Data = new Float32Array(int16Data.length);

        // Convert Int16 -> Float32
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

        const audioBuffer = ctx.createBuffer(1, chunk.length, 24000); // TTS is 24k
        audioBuffer.getChannelData(0).set(chunk);

        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);

        // Schedule
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
