'use client';

import { useVoiceAgent } from '@/hooks/useVoiceAgent';
import { useState, useEffect, useRef } from 'react';

export default function Home() {
  const [ttsProvider, setTtsProvider] = useState<'deepgram' | 'piper'>('deepgram');
  const [sttProvider, setSttProvider] = useState<'deepgram' | 'gladia'>('deepgram');
  const [sttLanguage, setSttLanguage] = useState<string>('english');

  // LLM Config
  const [llmProvider, setLlmProvider] = useState<'groq'>('groq');
  const [llmModel, setLlmModel] = useState<string>('llama-3.1-8b-instant');

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
    ttsSpeed,
    llmProvider,
    llmModel
  });

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcript]);

  return (
    <main className="relative flex h-screen w-screen flex-col bg-black text-white font-sans overflow-hidden selection:bg-orange-500/30">

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

      {/* Transcript Area (Top/Center) - Elastic Height */}
      <div className="flex-1 w-full max-w-3xl mx-auto pt-24 pb-32 px-6 flex flex-col justify-end min-h-0 z-10 relative">
        <div
          ref={scrollRef}
          className="flex flex-col gap-6 overflow-y-auto pr-2 mask-image-flow max-h-[80vh] [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-zinc-900/50 [&::-webkit-scrollbar-thumb]:bg-orange-500/50 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:hover:bg-orange-500"
          style={{ maskImage: 'linear-gradient(to bottom, transparent, black 10%, black)' }}
        >
          {transcript.length === 0 && (
            <div className="text-center text-zinc-600 mt-20 flex flex-col items-center gap-4 py-20">
              <div className="w-16 h-16 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center animate-pulse">
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
      <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50 flex flex-col items-center w-full max-w-lg px-4">

        {/* Settings Panel (Popup) */}
        {showSettings && (
          <div className="mb-4 w-full bg-zinc-900/95 backdrop-blur-xl border border-zinc-800 rounded-2xl p-4 shadow-2xl animate-in slide-in-from-bottom-5 fade-in duration-200 ring-1 ring-white/5">
            <div className="flex justify-between items-center mb-4 border-b border-zinc-800 pb-2">
              <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>
                Config
              </h3>
              <button onClick={() => setShowSettings(false)} className="text-zinc-500 hover:text-white transition-colors">&times;</button>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* STT */}
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Ears (STT)</label>
                <div className="relative">
                  <select
                    disabled={status === 'connected'}
                    value={sttProvider}
                    onChange={(e) => setSttProvider(e.target.value as any)}
                    className="w-full bg-black/40 border border-zinc-800 rounded-lg pl-3 pr-8 py-2.5 text-xs text-zinc-200 focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20 outline-none appearance-none transition-all disabled:opacity-50"
                  >
                    <option value="deepgram">Deepgram</option>
                    <option value="gladia">Gladia</option>
                  </select>
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-600">▼</div>
                </div>
              </div>

              {/* TTS */}
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Voice (TTS)</label>
                <div className="relative">
                  <select
                    disabled={status === 'connected'}
                    value={ttsProvider}
                    onChange={(e) => setTtsProvider(e.target.value as any)}
                    className="w-full bg-black/40 border border-zinc-800 rounded-lg pl-3 pr-8 py-2.5 text-xs text-zinc-200 focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20 outline-none appearance-none transition-all disabled:opacity-50"
                  >
                    <option value="deepgram">Deepgram Aura</option>
                    <option value="piper">Piper (Local)</option>
                  </select>
                  <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-600">▼</div>
                </div>
              </div>

              {/* Gladia Language - Conditional */}
              {sttProvider === 'gladia' && (
                <div className="space-y-1.5 col-span-2">
                  <label className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Language</label>
                  <div className="relative">
                    <select
                      disabled={status === 'connected'}
                      value={sttLanguage}
                      onChange={(e) => setSttLanguage(e.target.value)}
                      className="w-full bg-black/40 border border-zinc-800 rounded-lg pl-3 pr-8 py-2.5 text-xs text-zinc-200 focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20 outline-none appearance-none transition-all disabled:opacity-50"
                    >
                      <option value="english">English</option>
                      <option value="gujarati">Gujarati</option>
                      <option value="hindi">Hindi</option>
                    </select>
                    <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-600">▼</div>
                  </div>
                </div>
              )}

              {/* Piper Voice - Conditional */}
              {ttsProvider === 'piper' && (
                <>
                  <div className="space-y-1.5 col-span-2">
                    <label className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Model</label>
                    <div className="relative">
                      <select
                        disabled={status === 'connected'}
                        value={ttsVoice}
                        onChange={(e) => setTtsVoice(e.target.value)}
                        className="w-full bg-black/40 border border-zinc-800 rounded-lg pl-3 pr-8 py-2.5 text-xs text-zinc-200 focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20 outline-none appearance-none transition-all disabled:opacity-50"
                      >
                        {voices.map(v => <option key={v.id} value={v.id}>{v.name}</option>)}
                      </select>
                      <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-600">▼</div>
                    </div>
                  </div>

                  <div className="space-y-1.5 col-span-2">
                    <label className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Speed</label>
                    <div className="flex bg-black/40 rounded-lg border border-zinc-800 p-1">
                      {['super-slow', 'slow', 'normal', 'fast', 'super-fast'].map((s) => (
                        <button
                          key={s}
                          onClick={() => setTtsSpeed(s)}
                          disabled={status === 'connected'}
                          className={`flex-1 py-1.5 text-xs rounded-md transition-all font-medium ${ttsSpeed === s
                            ? 'bg-zinc-700 text-white shadow-sm'
                            : 'text-zinc-500 hover:text-zinc-300'
                            }`}
                        >
                          {s.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                        </button>
                      ))}
                    </div>
                  </div>
                </>
              )}

              {/* LLM Selection */}
              <div className="space-y-1.5 col-span-2 border-t border-zinc-800 pt-3 mt-1">
                <label className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wide">Brain (LLM)</label>
                <div className="grid grid-cols-2 gap-3">
                  <div className="relative">
                    <select
                      disabled={status === 'connected'}
                      value={llmProvider}
                      onChange={(e) => setLlmProvider(e.target.value as any)}
                      className="w-full bg-black/40 border border-zinc-800 rounded-lg pl-3 pr-8 py-2.5 text-xs text-zinc-200 focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20 outline-none appearance-none transition-all disabled:opacity-50"
                    >
                      <option value="groq">Groq</option>
                    </select>
                    <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-600">▼</div>
                  </div>
                  <div className="relative">
                    <select
                      disabled={status === 'connected'}
                      value={llmModel}
                      onChange={(e) => setLlmModel(e.target.value)}
                      className="w-full bg-black/40 border border-zinc-800 rounded-lg pl-3 pr-8 py-2.5 text-xs text-zinc-200 focus:border-orange-500/50 focus:ring-1 focus:ring-orange-500/20 outline-none appearance-none transition-all disabled:opacity-50"
                    >
                      <option value="llama-3.1-8b-instant">Llama 3.1 8B</option>
                      <option value="llama-3.1-70b-versatile">Llama 3.1 70B</option>
                      <option value="mixtral-8x7b-32768">Mixtral 8x7B</option>
                    </select>
                    <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-600">▼</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Floating Bar Container */}
        <div className="flex items-center gap-4 bg-zinc-900/90 backdrop-blur-xl border border-white/5 p-2 pl-4 rounded-full shadow-[0_20px_40px_-10px_rgba(0,0,0,0.8)] transition-all hover:border-white/10 hover:bg-zinc-900 ring-1 ring-black">

          {/* Settings Toggle */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`p-3 rounded-full transition-all duration-300 ${showSettings
              ? 'bg-zinc-800 text-white shadow-inner'
              : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-white'
              }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>
          </button>

          <div className="h-6 w-px bg-zinc-800"></div>

          {/* Main Call Button */}
          <button
            onClick={status === 'idle' || status === 'error' ? connect : disconnect}
            className={`
                group relative flex items-center justify-center gap-2 px-8 py-3.5 rounded-full font-bold transition-all duration-300 min-w-[160px] shadow-lg
                ${status === 'idle' || status === 'error'
                ? 'bg-zinc-100 text-black hover:bg-white hover:scale-105 hover:shadow-[0_0_30px_rgba(255,255,255,0.2)]'
                : 'bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 hover:border-red-500/50'
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
                <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                End Call
              </>
            )}
          </button>

          <div className="w-2"></div>

          {/* Mic Visualizer (Simple) */}
          <div className={`p-4 rounded-full transition-all duration-500 ${isSpeaking ? 'bg-orange-500 shadow-[0_0_30px_rgba(249,115,22,0.6)] scale-110' : 'bg-zinc-800 text-zinc-500'}`}>
            <span className="text-xl">
              {isSpeaking ? '🗣️' : '🎙️'}
            </span>
          </div>

        </div>
      </div>

    </main >
  );
}
