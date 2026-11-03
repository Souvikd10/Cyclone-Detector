"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Mic,
  MicOff,
  Send,
  Sparkles,
  Volume2,
  VolumeX,
  ChevronDown,
  Maximize2,
  Trash2,
  CheckCircle2,
  Radio,
  Layers,
  Clock,
  Bell,
  Gauge,
  HelpCircle,
  MapPin,
} from "lucide-react";
import type { Storm } from "@/types/storm";
import type {
  TranscriptTurn,
  AssistantMode,
  ToolExecutionRecord,
} from "@/types/voice";
import { fetchVoiceToken, sendVoiceChatMessage } from "@/lib/api";
import { PcmRecorder, PcmPlayer, speakText } from "@/lib/audioUtils";
import { processSimulatedTurn } from "@/lib/simulatedAgent";
import { GeminiLiveClient } from "@/lib/geminiLiveClient";
import { executeTool } from "@/lib/toolDeclarations";
import { useDashboardStore } from "@/store/useDashboardStore";

interface VoiceAssistantWidgetProps {
  storms: Storm[];
}

export default function VoiceAssistantWidget({ storms }: VoiceAssistantWidgetProps) {
  const { userLocation, setLocationModalOpen } = useDashboardStore();
  const [isOpen, setIsOpen] = useState(false);
  const [mode, setMode] = useState<AssistantMode>("simulated");
  const [token, setToken] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<string>("Gemini AI");
  const [isThinking, setIsThinking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [inputText, setInputText] = useState("");
  const [transcript, setTranscript] = useState<TranscriptTurn[]>([
    {
      id: "init",
      sender: "assistant",
      text: "Hello! I am your multilingual Cyclone Intelligence Assistant powered by Google Gemini. You can speak or type to me in English, Hindi (हिन्दी), or Bengali (বাংলা) to ask for cyclone weather updates, track forecasts, wind speed, landfall alerts, or control the dashboard map.",
      timestamp: Date.now(),
      language: "en",
    },
  ]);

  // Update initial greeting when userLocation is set
  useEffect(() => {
    if (userLocation) {
      setTranscript((prev) => {
        if (prev.length === 1 && prev[0].id === "init") {
          return [
            {
              id: "init",
              sender: "assistant",
              text: `Hello! I am your multilingual Cyclone Intelligence Assistant. Your location is verified as ${userLocation.city || "your area"} (${userLocation.lat.toFixed(2)}°N, ${userLocation.lon.toFixed(2)}°E). You can ask me how close the cyclone is to you, whether your area is in the danger zone, or check local weather alerts in English, Hindi (हिन्दी), or Bengali (বাংলা).`,
              timestamp: Date.now(),
              language: "en",
            },
          ];
        }
        return prev;
      });
    }
  }, [userLocation]);

  const [activeVoiceStatus, setActiveVoiceStatus] = useState<string>("Gemini AI Active");

  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  // Audio & Client references
  const liveClientRef = useRef<GeminiLiveClient | null>(null);
  const pcmPlayerRef = useRef<PcmPlayer | null>(null);
  const pcmRecorderRef = useRef<PcmRecorder | null>(null);
  const stopTtsRef = useRef<(() => void) | null>(null);
  const speechRecognitionRef = useRef<any>(null);

  // Initialize PCM Player on client side
  useEffect(() => {
    pcmPlayerRef.current = new PcmPlayer();
    return () => {
      pcmPlayerRef.current?.stop();
    };
  }, []);

  // Fetch ephemeral token from backend on mount
  useEffect(() => {
    async function initToken() {
      try {
        const tokenRes = await fetchVoiceToken();
        setToken(tokenRes.token);
        setMode(tokenRes.mode);

        if (tokenRes.mode === "live" && tokenRes.token) {
          // Setup Gemini Live WebSocket
          const client = new GeminiLiveClient(tokenRes.token, storms, {
            onConnected: () => {
              setActiveVoiceStatus("Connected to Gemini Live");
            },
            onDisconnected: () => {
              setActiveVoiceStatus("Disconnected");
            },
            onTextChunk: (chunk) => {
              appendAssistantChunk(chunk);
            },
            onAudioChunk: (base64Pcm) => {
              if (!isMuted) {
                setIsSpeaking(true);
                pcmPlayerRef.current?.playChunk(base64Pcm);
              }
            },
            onToolExecuted: (record) => {
              attachToolExecution(record);
            },
            onInterrupted: () => {
              handleBargeIn();
            },
            onError: (err) => {
              console.warn("[VoiceWidget] Gemini Live error:", err);
            },
          });
          client.connect();
          liveClientRef.current = client;
        } else {
          setActiveVoiceStatus("Simulated Mode (Awaiting API Key)");
        }
      } catch (err) {
        console.warn("[VoiceWidget] Failed initializing token:", err);
        setMode("simulated");
        setActiveVoiceStatus("Offline Simulated Mode");
      }
    }
    initToken();

    return () => {
      liveClientRef.current?.disconnect();
    };
  }, [storms]);

  // Auto-scroll transcript to bottom
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript]);

  // Barge-in: interrupt audio output when user starts typing or speaking
  const handleBargeIn = useCallback(() => {
    pcmPlayerRef.current?.stop();
    if (stopTtsRef.current) {
      stopTtsRef.current();
      stopTtsRef.current = null;
    }
    setIsSpeaking(false);
  }, []);

  // Append chunk to last assistant turn or create new turn
  const appendAssistantChunk = (chunk: string) => {
    setTranscript((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.sender === "assistant") {
        return [
          ...prev.slice(0, -1),
          { ...last, text: last.text + chunk, timestamp: Date.now() },
        ];
      }
      return [
        ...prev,
        {
          id: "turn_" + Date.now(),
          sender: "assistant",
          text: chunk,
          timestamp: Date.now(),
        },
      ];
    });
  };

  // Attach a tool execution record to the most recent assistant turn
  const attachToolExecution = (record: ToolExecutionRecord) => {
    setTranscript((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.sender === "assistant") {
        const existing = last.toolExecutions || [];
        return [
          ...prev.slice(0, -1),
          { ...last, toolExecutions: [...existing, record] },
        ];
      }
      return [
        ...prev,
        {
          id: "turn_" + Date.now(),
          sender: "assistant",
          text: "",
          timestamp: Date.now(),
          toolExecutions: [record],
        },
      ];
    });
  };

  // Send turn (either text or voice transcription)
  const handleSendTurn = async (text: string, inputMethod: "text" | "voice" = "text") => {
    if (!text.trim()) return;
    handleBargeIn();

    const userTurn: TranscriptTurn = {
      id: "user_" + Date.now(),
      sender: "user",
      text: text.trim(),
      inputMethod,
      timestamp: Date.now(),
    };

    setTranscript((prev) => [...prev, userTurn]);
    setInputText("");

    setIsThinking(true);

    try {
      // 1. Gather conversation history for Gemini context
      const chatHistory = transcript
        .filter((t) => t.text && t.id !== "init" && t.sender !== "system")
        .slice(-6)
        .map((t) => ({
          role: (t.sender === "user" ? "user" : "assistant") as "user" | "assistant",
          content: t.text,
        }));

      const currentStormId = useDashboardStore.getState().selectedStormId;
      const userLoc = useDashboardStore.getState().userLocation;

      // 2. Call backend Gemini conversational cyclone intelligence endpoint
      const geminiRes = await sendVoiceChatMessage({
        message: text.trim(),
        history: chatHistory,
        current_storm_id: currentStormId,
        user_location: userLoc
          ? {
              lat: userLoc.lat,
              lon: userLoc.lon,
              city: userLoc.city,
              state: userLoc.state,
              country: userLoc.country,
            }
          : null,
      });

      if (geminiRes.model) {
        setActiveModel(geminiRes.model.replace("models/", ""));
      }

      // 3. Execute any tool / UI actions returned by Gemini
      const executedTools: ToolExecutionRecord[] = [];
      if (geminiRes.tool_executions && geminiRes.tool_executions.length > 0) {
        for (const item of geminiRes.tool_executions) {
          const res = await executeTool(item.name, item.args, storms);
          executedTools.push({
            id: item.id || "call_" + Math.random().toString(36).slice(2, 8),
            name: item.name as any,
            args: item.args,
            result: res,
            timestamp: Date.now(),
          });
        }
      }

      const assistantTurn: TranscriptTurn = {
        id: "assistant_" + Date.now(),
        sender: "assistant",
        text: geminiRes.text,
        timestamp: Date.now(),
        language: (geminiRes.language as any) || "en",
        toolExecutions: executedTools,
      };

      setTranscript((prev) => [...prev, assistantTurn]);
      setActiveVoiceStatus(`Gemini Active (${geminiRes.model || "Flash"})`);

      // 4. Speak response aloud if unmuted
      if (!isMuted && geminiRes.text) {
        setIsSpeaking(true);
        stopTtsRef.current = speakText(
          geminiRes.text,
          (geminiRes.language as any) || "en",
          () => {
            setIsSpeaking(false);
            stopTtsRef.current = null;
          }
        );
      }
    } catch (err) {
      console.warn("[VoiceWidget] Gemini endpoint unreachable, falling back to local simulated turn:", err);
      const userLoc = useDashboardStore.getState().userLocation;
      const response = await processSimulatedTurn(text.trim(), storms, userLoc);

      const assistantTurn: TranscriptTurn = {
        id: "assistant_" + Date.now(),
        sender: "assistant",
        text: response.text,
        timestamp: Date.now(),
        language: response.language,
        toolExecutions: response.toolExecutions,
      };

      setTranscript((prev) => [...prev, assistantTurn]);
      setActiveVoiceStatus("Offline Fallback Mode");

      if (!isMuted) {
        setIsSpeaking(true);
        stopTtsRef.current = speakText(response.text, response.language, () => {
          setIsSpeaking(false);
          stopTtsRef.current = null;
        });
      }
    } finally {
      setIsThinking(false);
    }
  };


  // ── Voice Mic Toggle ──
  const toggleListening = async () => {
    handleBargeIn();

    if (isListening) {
      // Stop recording
      setIsListening(false);
      pcmRecorderRef.current?.stop();
      if (speechRecognitionRef.current) {
        speechRecognitionRef.current.stop();
        speechRecognitionRef.current = null;
      }
      return;
    }

    // Start recording
    try {
      setIsListening(true);

      if (mode === "live" && liveClientRef.current?.connected) {
        // Live API streaming
        const recorder = new PcmRecorder((chunk) => {
          liveClientRef.current?.sendAudioChunk(chunk);
        });
        await recorder.start();
        pcmRecorderRef.current = recorder;
      } else {
        // Simulated mode using browser Web Speech recognition
        const SpeechRec =
          (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

        if (SpeechRec) {
          const rec = new SpeechRec();
          rec.continuous = false;
          rec.interimResults = true;
          rec.lang = "en-IN"; // also recognizes multilingual or transliterated

          rec.onresult = (e: any) => {
            const transcriptText = Array.from(e.results)
              .map((res: any) => res[0].transcript)
              .join("");

            if (e.results[0].isFinal) {
              setIsListening(false);
              handleSendTurn(transcriptText, "voice");
            }
          };

          rec.onerror = (e: any) => {
            console.warn("[VoiceWidget] Speech recognition notice:", e.error);
            setIsListening(false);
          };

          rec.onend = () => {
            setIsListening(false);
          };

          speechRecognitionRef.current = rec;
          rec.start();
        } else {
          // Fallback prompt if SpeechRecognition is blocked
          setTimeout(() => {
            setIsListening(false);
            handleSendTurn("Is my area in danger?", "voice");
          }, 1500);
        }
      }
    } catch (err) {
      console.warn("[VoiceWidget] Mic permission error:", err);
      setIsListening(false);
    }
  };

  const clearTranscript = () => {
    handleBargeIn();
    setTranscript([
      {
        id: "cleared_" + Date.now(),
        sender: "system",
        text: "Conversation session cleared.",
        timestamp: Date.now(),
      },
    ]);
  };

  return (
    <>
      {/* ── Collapsed Floating Button ── */}
      {!isOpen && (
        <div className="fixed bottom-5 right-5 z-50 animate-bounce-subtle">
          <button
            onClick={() => setIsOpen(true)}
            className="group relative flex items-center gap-3 px-4 py-3 rounded-full bg-gradient-to-r from-blue-600 via-indigo-600 to-cyan-500 text-white shadow-2xl shadow-blue-500/40 hover:shadow-cyan-500/60 hover:scale-105 transition-all duration-300 border border-white/20"
          >
            {/* Pulsing beacon */}
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-300 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-400"></span>
            </span>

            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-200 animate-pulse" />
              <span className="text-xs font-semibold tracking-wide">
                Voice + Text AI
              </span>
              <span className="text-[0.65rem] px-1.5 py-0.5 rounded bg-white/20 font-mono">
                EN · HI · BN
              </span>
            </div>

            <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center group-hover:bg-white/30 transition-colors">
              <Mic className="w-4 h-4 text-white" />
            </div>
          </button>
        </div>
      )}

      {/* ── Expanded Conversational Window ── */}
      {isOpen && (
        <div className="fixed bottom-5 right-5 z-50 w-[95vw] sm:w-[420px] h-[580px] max-h-[85vh] flex flex-col glass-card border border-[var(--bg-glass-border)] shadow-2xl shadow-black/40 rounded-2xl overflow-hidden animate-slide-in">
          {/* Header */}
          <div className="p-3.5 border-b border-[var(--bg-glass-border)] bg-[var(--bg-secondary)] flex items-center justify-between">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-blue-500 to-cyan-400 flex items-center justify-center flex-shrink-0 shadow-md shadow-blue-500/30">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <h3 className="text-xs font-bold text-[var(--text-primary)] truncate">
                    Cyclone AI Assistant
                  </h3>
                  <span
                    className="badge text-[0.62rem] py-0.5 px-2 bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    {activeModel ? `Gemini AI (${activeModel})` : "Gemini AI Active"}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-[0.68rem] text-[var(--text-muted)]">
                  <span>English</span> • <span>हिन्दी</span> • <span>বাংলা</span>
                </div>
              </div>
            </div>

            {/* Header controls */}
            <div className="flex items-center gap-1.5">
              {/* Location Pill in Widget Header */}
              <button
                onClick={() => setLocationModalOpen(true)}
                title={userLocation ? "Click to change verified location" : "Enable location services"}
                className={`flex items-center gap-1 text-[0.68rem] px-2 py-0.5 rounded-full border transition-all ${
                  userLocation
                    ? "bg-cyan-500/10 border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/20"
                    : "bg-amber-500/15 border-amber-500/30 text-amber-300 hover:bg-amber-500/25 animate-pulse"
                }`}
              >
                <MapPin className="w-3 h-3 text-cyan-400" />
                <span className="truncate max-w-[80px]">
                  {userLocation ? userLocation.city || "Location Set" : "Set Location"}
                </span>
              </button>

              <button
                onClick={() => {
                  handleBargeIn();
                  setIsMuted(!isMuted);
                }}
                title={isMuted ? "Unmute Spoken Replies" : "Mute Spoken Replies"}
                className="btn-ghost p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)]"
              >
                {isMuted ? (
                  <VolumeX className="w-3.5 h-3.5 text-rose-400" />
                ) : (
                  <Volume2 className="w-3.5 h-3.5 text-cyan-400" />
                )}
              </button>

              <button
                onClick={clearTranscript}
                title="Clear Transcript"
                className="btn-ghost p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)]"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>

              <button
                onClick={() => {
                  handleBargeIn();
                  setIsOpen(false);
                }}
                title="Minimize Widget"
                className="btn-ghost p-1.5 rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)]"
              >
                <ChevronDown className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Quick Action Chips */}
          <div className="px-3 py-1.5 bg-[var(--bg-primary)]/70 border-b border-[var(--bg-glass-border)] flex items-center gap-1.5 overflow-x-auto custom-scrollbar flex-shrink-0 text-[0.68rem]">
            <span className="text-[var(--text-muted)] whitespace-nowrap text-[0.62rem]">Try:</span>
            <button
              onClick={() => handleSendTurn("Is my area in danger?")}
              className="px-2 py-0.5 rounded-full bg-cyan-500/15 border border-cyan-500/30 hover:border-cyan-400 text-cyan-300 whitespace-nowrap font-medium"
            >
              📍 My Area Threat
            </button>
            <button
              onClick={() => handleSendTurn("मेरे इलाके में क्या खतरा है?")}
              className="px-2 py-0.5 rounded-full bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] hover:border-blue-500/50 text-[var(--text-secondary)] whitespace-nowrap"
            >
              खतरा जांचें (हिन्दी)
            </button>
            <button
              onClick={() => handleSendTurn("আমাদের এলাকায় কি কোনো বিপদ আছে?")}
              className="px-2 py-0.5 rounded-full bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] hover:border-blue-500/50 text-[var(--text-secondary)] whitespace-nowrap"
            >
              বিপদ নিরীক্ষা (বাংলা)
            </button>
            <button
              onClick={() => handleSendTurn("Basin status summary")}
              className="px-2 py-0.5 rounded-full bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] hover:border-blue-500/50 text-[var(--text-secondary)] whitespace-nowrap"
            >
              Basin Status
            </button>
            <button
              onClick={() => handleSendTurn("Toggle forecast cone layer")}
              className="px-2 py-0.5 rounded-full bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] hover:border-blue-500/50 text-[var(--text-secondary)] whitespace-nowrap"
            >
              Toggle Cone
            </button>
            <button
              onClick={() => handleSendTurn("Open alert panel")}
              className="px-2 py-0.5 rounded-full bg-[var(--bg-secondary)] border border-[var(--bg-glass-border)] hover:border-blue-500/50 text-[var(--text-secondary)] whitespace-nowrap"
            >
              Open Alerts
            </button>
            {storms.length > 0 && (
              <button
                onClick={() => handleSendTurn(`Show me ${storms[0].name}`)}
                className="px-2 py-0.5 rounded-full bg-blue-500/20 border border-blue-500/40 text-blue-300 whitespace-nowrap font-medium"
              >
                Track {storms[0].name}
              </button>
            )}
          </div>

          {/* Transcript Feed */}
          <div className="flex-1 p-3.5 overflow-y-auto space-y-3 custom-scrollbar">
            {transcript.map((turn) => {
              if (turn.sender === "system") {
                return (
                  <div key={turn.id} className="text-center my-2">
                    <span className="text-[0.68rem] text-[var(--text-muted)] bg-[var(--bg-secondary)] px-2.5 py-1 rounded-full border border-[var(--bg-glass-border)]">
                      {turn.text}
                    </span>
                  </div>
                );
              }

              const isUser = turn.sender === "user";

              return (
                <div
                  key={turn.id}
                  className={`flex flex-col ${isUser ? "items-end" : "items-start"} gap-1`}
                >
                  <div className="flex items-center gap-1.5 text-[0.65rem] text-[var(--text-muted)] px-1">
                    <span>{isUser ? "You" : "Cyclone AI"}</span>
                    {turn.inputMethod === "voice" && (
                      <span className="flex items-center gap-0.5 text-blue-400">
                        <Mic className="w-2.5 h-2.5" /> Voice
                      </span>
                    )}
                    {turn.inputMethod === "text" && (
                      <span className="text-[var(--text-muted)]">⌨️ Typed</span>
                    )}
                  </div>

                  <div
                    className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-xs leading-relaxed shadow-sm ${
                      isUser
                        ? "bg-blue-600 text-white rounded-br-xs"
                        : "bg-[var(--bg-secondary)] text-[var(--text-primary)] border border-[var(--bg-glass-border)] rounded-bl-xs"
                    }`}
                  >
                    {turn.text}

                    {/* Tool Executions Badge */}
                    {turn.toolExecutions && turn.toolExecutions.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-[var(--bg-glass-border)] space-y-1.5">
                        {turn.toolExecutions.map((exec) => (
                          <div
                            key={exec.id}
                            className="flex items-start gap-1.5 p-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[0.7rem]"
                          >
                            <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-emerald-400" />
                            <div>
                              <span className="font-semibold uppercase tracking-wider text-[0.62rem] block">
                                Executed: {exec.name}
                              </span>
                              <span className="text-[0.68rem] text-[var(--text-secondary)]">
                                {formatToolResult(exec.name, exec.result)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {isThinking && (
              <div className="flex items-center gap-2 p-2.5 rounded-xl bg-blue-500/10 border border-blue-500/20 text-cyan-300 text-xs animate-pulse">
                <Sparkles className="w-3.5 h-3.5 animate-spin text-cyan-400 flex-shrink-0" />
                <span>Gemini AI is analyzing cyclone data...</span>
              </div>
            )}

            {isSpeaking && (
              <div className="flex items-center gap-2 text-[0.7rem] text-cyan-400 animate-pulse px-2">
                <Radio className="w-3.5 h-3.5 animate-spin" />
                <span>Assistant is speaking... (tap mic or type to barge in)</span>
              </div>
            )}

            <div ref={transcriptEndRef} />
          </div>

          {/* Listening Indicator Bar */}
          {isListening && (
            <div className="px-3 py-2 bg-blue-500/10 border-t border-blue-500/20 flex items-center justify-between text-xs text-blue-400">
              <div className="flex items-center gap-2">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-red-500"></span>
                </span>
                <span className="font-medium animate-pulse">Listening to microphone...</span>
              </div>
              <span className="text-[0.68rem] text-[var(--text-muted)]">Speak in English, Hindi, or Bengali</span>
            </div>
          )}

          {/* Quick Location & Cyclone Suggestion Chips */}
          <div className="px-3 py-1.5 bg-[var(--bg-secondary)] border-t border-[var(--bg-glass-border)] flex items-center gap-1.5 overflow-x-auto custom-scrollbar">
            <button
              onClick={() => handleSendTurn("How far is the cyclone from my area?", "text")}
              disabled={isThinking}
              className="text-[0.68rem] px-2 py-1 rounded-lg bg-[var(--bg-primary)] border border-[var(--bg-glass-border)] text-[var(--text-secondary)] hover:text-cyan-300 hover:border-cyan-500/40 whitespace-nowrap transition-all flex items-center gap-1 cursor-pointer disabled:opacity-50"
            >
              <MapPin className="w-2.5 h-2.5 text-cyan-400" />
              <span>Distance to my area</span>
            </button>
            <button
              onClick={() => handleSendTurn("Am I in the danger zone?", "text")}
              disabled={isThinking}
              className="text-[0.68rem] px-2 py-1 rounded-lg bg-[var(--bg-primary)] border border-[var(--bg-glass-border)] text-[var(--text-secondary)] hover:text-amber-300 hover:border-amber-500/40 whitespace-nowrap transition-all flex items-center gap-1 cursor-pointer disabled:opacity-50"
            >
              <span>⚠️ Danger zone risk?</span>
            </button>
            <button
              onClick={() => handleSendTurn("What is the wind speed and rain alert here?", "text")}
              disabled={isThinking}
              className="text-[0.68rem] px-2 py-1 rounded-lg bg-[var(--bg-primary)] border border-[var(--bg-glass-border)] text-[var(--text-secondary)] hover:text-blue-300 hover:border-blue-500/40 whitespace-nowrap transition-all flex items-center gap-1 cursor-pointer disabled:opacity-50"
            >
              <span>🌧️ Local wind & rain</span>
            </button>
            <button
              onClick={() => handleSendTurn("Which cyclone is closest to me?", "text")}
              disabled={isThinking}
              className="text-[0.68rem] px-2 py-1 rounded-lg bg-[var(--bg-primary)] border border-[var(--bg-glass-border)] text-[var(--text-secondary)] hover:text-emerald-300 hover:border-emerald-500/40 whitespace-nowrap transition-all flex items-center gap-1 cursor-pointer disabled:opacity-50"
            >
              <span>🗺️ Closest cyclone</span>
            </button>
          </div>

          {/* Input Controls Bar */}
          <div className="p-3 border-t border-[var(--bg-glass-border)] bg-[var(--bg-secondary)] flex items-center gap-2">
            {/* Mic Toggle Button */}
            <button
              onClick={toggleListening}
              title={isListening ? "Stop Listening" : "Start Voice Input (EN/HI/BN)"}
              className={`p-2.5 rounded-xl flex items-center justify-center transition-all ${
                isListening
                  ? "bg-red-500 text-white shadow-lg shadow-red-500/50 animate-pulse"
                  : "bg-[var(--bg-primary)] text-[var(--text-secondary)] hover:text-white hover:bg-blue-600 border border-[var(--bg-glass-border)]"
              }`}
            >
              {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            </button>

            {/* Text Input */}
            <div className="flex-1 relative flex items-center">
              <input
                type="text"
                value={inputText}
                disabled={isThinking}
                onChange={(e) => {
                  handleBargeIn();
                  setInputText(e.target.value);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !isThinking) {
                    e.preventDefault();
                    handleSendTurn(inputText, "text");
                  }
                }}
                placeholder={isThinking ? "Gemini is replying..." : "Type or speak (English, हिन्दी, বাংলা)..."}
                className="w-full px-3 py-2 rounded-xl bg-[var(--bg-primary)] border border-[var(--bg-glass-border)] text-xs text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-blue-500/50 transition-all pr-8 disabled:opacity-60"
              />
            </div>

            {/* Send Button */}
            <button
              onClick={() => handleSendTurn(inputText, "text")}
              disabled={!inputText.trim() || isThinking}
              className="p-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-all shadow-md shadow-blue-600/30"
            >
              {isThinking ? (
                <Sparkles className="w-4 h-4 animate-spin text-white" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>

        </div>
      )}
    </>
  );
}

function formatToolResult(name: string, result: Record<string, unknown>): string {
  if (!result) return "Success";

  if (name === "navigate_to_storm") {
    return `Navigated to ${result.name || "storm"} (${result.category || "Active"})`;
  }
  if (name === "toggle_map_layer") {
    return `Set layer '${result.layer}' to ${result.visible ? "Visible" : "Hidden"}`;
  }
  if (name === "start_replay") {
    return `Started replay timeline for ${result.storm_name || "storm"}`;
  }
  if (name === "open_alert_panel") {
    return `Opened CAP 1.2 emergency advisory`;
  }
  if (name === "get_storm_status") {
    return `${result.storm_name}: ${result.category}, ${result.max_wind_speed_kts} kts, RI: ${Math.round(
      Number(result.ri_probability || 0) * 100
    )}%`;
  }

  return JSON.stringify(result);
}
