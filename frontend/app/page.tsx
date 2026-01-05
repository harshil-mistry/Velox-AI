'use client';

import { useVoiceAgent } from '@/hooks/useVoiceAgent';
import { useState, useEffect, useRef } from 'react';

export default function Home() {
  const [ttsProvider, setTtsProvider] = useState<'deepgram' | 'piper'>('deepgram');
  const [sttProvider, setSttProvider] = useState<'deepgram' | 'gladia'>('deepgram');
  const [sttLanguage, setSttLanguage] = useState<string>('english');

  // Determine WS URL dynamically (client-side only)
  const [wsUrl, setWsUrl] = useState('');

  useEffect(() => {
    // This runs only on the client
    setWsUrl(`ws://${window.location.hostname}:8000/ws/stream`);
  }, []);

  const { status, isSpeaking, transcript, connect, disconnect } = useVoiceAgent(
    wsUrl,
    'velox-secret-123',
    ttsProvider,
    sttProvider,
    sttLanguage
  );

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcript]);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gray-900 text-white">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
          Velox AI Dashboard
        </p>
        <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify-center bg-gradient-to-t from-white via-white dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
          <div className="pointer-events-none flex place-items-center gap-2 p-8 lg:pointer-events-auto lg:p-0">
            Status: <span className={`font-bold ${status === 'connected' ? 'text-green-500' : 'text-red-500'}`}>{status}</span>
          </div>
        </div>
      </div>

      <div className="relative flex flex-col items-center justify-center py-10">

        {/* Settings Panel */}
        <div className="absolute top-0 right-0 p-4 flex flex-col gap-2 bg-gray-900 rounded-bl-xl border-l border-b border-gray-700">

          {/* STT Selector */}
          <div>
            <label className="block text-xs font-bold text-gray-400 mb-1">STT Provider</label>
            <select
              disabled={status === 'connected'}
              value={sttProvider}
              onChange={(e) => setSttProvider(e.target.value as 'deepgram' | 'gladia')}
              className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-blue-500 disabled:opacity-50"
            >
              <option value="deepgram">Deepgram Nova-2</option>
              <option value="gladia">Gladia (Cloud)</option>
            </select>
          </div>

          {/* Language Selector (Gladia Only) */}
          {sttProvider === 'gladia' && (
            <div>
              <label className="block text-xs font-bold text-gray-400 mb-1">Language</label>
              <select
                disabled={status === 'connected'}
                value={sttLanguage}
                onChange={(e) => setSttLanguage(e.target.value)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-blue-500 disabled:opacity-50"
              >
                <option value="english">English (default)</option>
                <option value="gujarati">Gujarati</option>
                <option value="hindi">Hindi</option>
              </select>
            </div>
          )}

          {/* TTS Selector */}
          <div>
            <label className="block text-xs font-bold text-gray-400 mb-1">TTS Provider</label>
            <select
              disabled={status === 'connected'}
              value={ttsProvider}
              onChange={(e) => setTtsProvider(e.target.value as 'deepgram' | 'piper')}
              className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-white focus:outline-none focus:border-blue-500 disabled:opacity-50"
            >
              <option value="deepgram">Deepgram Aura (Cloud)</option>
              <option value="piper">Piper (Local)</option>
            </select>
          </div>
        </div>

        <div className={`w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 ${isSpeaking ? 'bg-green-500 animate-pulse scale-110' : 'bg-gray-700'}`}>
          {isSpeaking ? '🗣️' : '👂'}
        </div>

        <div className="mt-10 flex gap-4">
          {status === 'idle' || status === 'error' ? (
            <button
              onClick={connect}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-bold transition-colors"
            >
              Start Call
            </button>
          ) : (
            <button
              onClick={disconnect}
              className="px-6 py-3 bg-red-600 hover:bg-red-700 rounded-lg font-bold transition-colors"
            >
              End Call
            </button>
          )}
        </div>
      </div>

      <div className="mt-8 w-full max-w-2xl bg-gray-800 rounded-xl p-6 h-96 overflow-y-auto border border-gray-700" ref={scrollRef}>
        {transcript.length === 0 && <p className="text-gray-500 text-center">No transcript yet...</p>}
        {transcript.map((msg, idx) => (
          <div key={idx} className={`mb-4 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
            <span className={`inline-block px-4 py-2 rounded-lg ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-200'}`}>
              {msg.content}
            </span>
          </div>
        ))}
      </div>
    </main>
  );
}
