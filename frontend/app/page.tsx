'use client';

import { useVoiceAgent } from '@/hooks/useVoiceAgent';
import { useState, useEffect, useRef } from 'react';

export default function Home() {
  const [ttsProvider, setTtsProvider] = useState<'deepgram' | 'piper'>('deepgram');
  const [sttProvider, setSttProvider] = useState<'deepgram' | 'gladia'>('deepgram');
  const [sttLanguage, setSttLanguage] = useState<string>('english');

  // Piper Config
  const [voices, setVoices] = useState<{ id: string, name: string }[]>([]);
  const [ttsVoice, setTtsVoice] = useState<string>('');
  const [ttsSpeed, setTtsSpeed] = useState<string>('normal');

  // UI State
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    // Fetch voices on mount (client-side)
    fetch(`http://${window.location.hostname}:8000/voices`)
      .then(res => res.json())
      .then(data => {
        setVoices(data);
        if (data.length > 0) setTtsVoice(data[0].id);
      })
      .catch(err => console.error("Failed to fetch voices", err));
  }, []);

  // Determine WS URL dynamically (client-side only)
  const [wsUrl, setWsUrl] = useState('');

  useEffect(() => {
    setWsUrl(`ws://${window.location.hostname}:8000/ws/stream`);
  }, []);

  const { status, isSpeaking, transcript, connect, disconnect } = useVoiceAgent({
    serverUrl: wsUrl,
    token: 'velox-secret-123',
    ttsProvider,
    ttsVoice,
    sttProvider,
    sttLanguage,
    ttsSpeed
  });

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcript]);

  return (
    <main className="relative flex min-h-screen flex-col bg-black text-white selection:bg-orange-500/30 font-sans overflow-hidden">

      {/* Header */}
      <header className="absolute top-0 left-0 w-full p-6 z-20 flex justify-between items-center pointer-events-none">
        <div className="flex items-center gap-2 pointer-events-auto">
          <div className="w-3 h-3 rounded-full bg-orange-500 shadow-[0_0_10px_rgba(249,115,22,0.6)]" />
          <h1 className="text-xl font-bold tracking-tight text-white/90">Velox<span className="text-orange-500">AI</span></h1>
        </div>

        {/* Connection Status Badge */}
        <div className={`px-3 py-1 rounded-full text-xs font-medium border pointer-events-auto transition-colors duration-300 ${status === 'connected'
            ? 'bg-green-500/10 border-green-500/20 text-green-400'
            : status === 'error'
              ? 'bg-red-500/10 border-red-500/20 text-red-400'
              : 'bg-zinc-800/50 border-zinc-700 text-zinc-400'
          }`}>
          {status === 'idle' ? 'Ready' : status.charAt(0).toUpperCase() + status.slice(1)}
        </div>
      </header>

      {/* Transcript Area (Top/Center) */}
      <div className="flex-1 w-full max-w-3xl mx-auto pt-24 pb-48 px-6 flex flex-col justify-end min-h-0 z-10">
        <div
          ref={scrollRef}
          className="flex flex-col gap-6 overflow-y-auto pr-2 scrollbar-hide mask-image-flow"
          style={{ maskImage: 'linear-gradient(to bottom, transparent, black 10%, black)' }}
        >
          {transcript.length === 0 && (
            <div className="text-center text-zinc-600 mt-20 flex flex-col items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center">
                <span className="text-2xl opacity-50">✨</span>
              </div>
              <p>Start conversation to begin...</p>
            </div>
          )}

          {transcript.map((msg, idx) => (
            <div
              key={idx}
              className={`flex flex-col max-w-[85%] ${msg.role === 'user' ? 'self-end items-end' : 'self-start items-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}
            >
              <div
                className={`px-5 py-3 rounded-2xl text-[15px] leading-relaxed shadow-sm transition-all ${msg.role === 'user'
                    ? 'bg-zinc-800 text-zinc-100 rounded-tr-sm border border-zinc-700'
                    : 'bg-orange-950/20 text-orange-100 rounded-tl-sm border border-orange-500/20 shadow-[0_4px_20px_-10px_rgba(249,115,22,0.2)]'
                  }`}
              >
                {msg.content}
              </div>
              <span className="text-[10px] text-zinc-600 mt-1 px-1">
                {msg.role === 'user' ? 'You' : 'Assistant'}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Floating Navbar (Bottom) */}
      <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50 flex flex-col items-center w-full max-w-lg">

        {/* Settings Panel (Popup) */}
        {showSettings && (
          <div className="mb-4 w-[90%] bg-zinc-900/90 backdrop-blur-xl border border-zinc-800 rounded-2xl p-4 shadow-2xl animate-in slide-in-from-bottom-5 fade-in duration-200">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-wider">Settings</h3>
              <button onClick={() => setShowSettings(false)} className="text-zinc-500 hover:text-white">&times;</button>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* STT */}
              <div className="space-y-1">
                <label className="text-xs text-zinc-500">Hearing (STT)</label>
                <select
                  disabled={status === 'connected'}
                  value={sttProvider}
                  onChange={(e) => setSttProvider(e.target.value as any)}
                  className="w-full bg-black/50 border border-zinc-700 rounded-lg px-2 py-2 text-sm text-white focus:border-orange-500 outline-none"
                >
                  <option value="deepgram">Deepgram</option>
                  <option value="gladia">Gladia</option>
                </select>
              </div>

              {/* TTS */}
              <div className="space-y-1">
                <label className="text-xs text-zinc-500">Voice (TTS)</label>
                <select
                  disabled={status === 'connected'}
                  value={ttsProvider}
                  onChange={(e) => setTtsProvider(e.target.value as any)}
                  className="w-full bg-black/50 border border-zinc-700 rounded-lg px-2 py-2 text-sm text-white focus:border-orange-500 outline-none"
                >
                  <option value="deepgram">Deepgram Aura</option>
                  <option value="piper">Piper (Local)</option>
                </select>
              </div>

              {/* Gladia Language - Conditional */}
              {sttProvider === 'gladia' && (
                <div className="space-y-1 col-span-2">
                  <label className="text-xs text-zinc-500">Data Language</label>
                  <select
                    disabled={status === 'connected'}
                    value={sttLanguage}
                    onChange={(e) => setSttLanguage(e.target.value)}
                    className="w-full bg-black/50 border border-zinc-700 rounded-lg px-2 py-2 text-sm text-white focus:border-orange-500 outline-none"
                  >
                    <option value="english">English</option>
                    <option value="gujarati">Gujarati</option>
                    <option value="hindi">Hindi</option>
                  </select>
                </div>
              )}

              {/* Piper Voice - Conditional */}
              {ttsProvider === 'piper' && (
                <>
                  <div className="space-y-1 col-span-2">
                    <label className="text-xs text-zinc-500">Model</label>
                    <select
                      disabled={status === 'connected'}
                      value={ttsVoice}
                      onChange={(e) => setTtsVoice(e.target.value)}
                      className="w-full bg-black/50 border border-zinc-700 rounded-lg px-2 py-2 text-sm text-white focus:border-orange-500 outline-none"
                    >
                      {voices.map(v => <option key={v.id} value={v.id}>{v.name}</option>)}
                    </select>
                  </div>

                  <div className="space-y-1 col-span-2">
                    <label className="text-xs text-zinc-500">Speed</label>
                    <div className="flex bg-black/50 rounded-lg border border-zinc-700 p-1">
                      {['slow', 'normal', 'fast'].map((s) => (
                        <button
                          key={s}
                          onClick={() => setTtsSpeed(s)}
                          disabled={status === 'connected'}
                          className={`flex-1 py-1 text-xs rounded transition-all ${ttsSpeed === s
                              ? 'bg-zinc-700 text-white shadow-sm'
                              : 'text-zinc-500 hover:text-zinc-300'
                            }`}
                        >
                          {s.charAt(0).toUpperCase() + s.slice(1)}
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Floating Bar Container */}
        <div className="flex items-center gap-4 bg-zinc-900/80 backdrop-blur-xl border border-zinc-800/50 p-2 pl-4 rounded-full shadow-[0_0_40px_-10px_rgba(0,0,0,0.5)] transition-all hover:border-zinc-700 hover:bg-zinc-900">

          {/* Settings Toggle */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-3 rounded-full transition-all ${showSettings ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-white'}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.72v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" /><circle cx="12" cy="12" r="3" /></svg>
          </button>

          <div className="h-6 w-px bg-zinc-800"></div>

          {/* Main Call Button */}
          <button
            onClick={status === 'idle' || status === 'error' ? connect : disconnect}
            className={`
                group relative flex items-center justify-center gap-2 px-6 py-3 rounded-full font-bold transition-all duration-300 min-w-[140px]
                ${status === 'idle' || status === 'error'
                ? 'bg-zinc-100 text-black hover:bg-white hover:shadow-[0_0_20px_rgba(255,255,255,0.3)]'
                : 'bg-red-500/10 text-red-500 border border-red-500/50 hover:bg-red-500/20'
              }
              `}
          >
            {status === 'idle' || status === 'error' ? (
              <>
                <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                Start Call
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-red-500"></span>
                End Call
              </>
            )}
          </button>

          <div className="w-2"></div>

          {/* Mic Visualizer (Simple) */}
          <div className={`p-4 rounded-full transition-all duration-300 ${isSpeaking ? 'bg-orange-500 shadow-[0_0_20px_rgba(249,115,22,0.5)] transform scale-110' : 'bg-zinc-800'}`}>
            <a className="text-xl">
              {isSpeaking ? '🗣️' : '🎙️'}
            </a>
          </div>

        </div>
      </div>

    </main>
  );
}
