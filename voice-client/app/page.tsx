"use client";

import { useEffect, useRef, useState } from "react";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

// Fix missing SpeechRecognition types in TS

interface SpeechRecognition extends EventTarget {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onaudiostart: ((this: SpeechRecognition, ev: Event) => any) | null;
  onsoundstart: ((this: SpeechRecognition, ev: Event) => any) | null;
  onspeechstart: ((this: SpeechRecognition, ev: Event) => any) | null;
  onspeechend: ((this: SpeechRecognition, ev: Event) => any) | null;
  onsoundend: ((this: SpeechRecognition, ev: Event) => any) | null;
  onaudioend: ((this: SpeechRecognition, ev: Event) => any) | null;
  onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
  onnomatch: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
  onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => any) | null;
  onstart: ((this: SpeechRecognition, ev: Event) => any) | null;
  onend: ((this: SpeechRecognition, ev: Event) => any) | null;
}

interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface Window {
  webkitSpeechRecognition: any;
  SpeechRecognition: any;
}


const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export default function Home() {
  const [conversationId, setConversationId] = useState("demo_conv_1");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const recognitionRef = useRef<SpeechRecognition | null>(null);

  // ----------- Helpers: STT + TTS -----------

  // Initialize speech recognition
  const initRecognition = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setError("Browser STT not supported. Use latest Chrome / Edge.");
      return null;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setIsListening(true);
      setError(null);
    };

    recognition.onerror = (event: any) => {
      console.error("STT error:", event.error);
      setError(`STT error: ${event.error}`);
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = event.results[0][0].transcript.trim();
      console.log("Heard:", transcript);
      if (transcript) {
        handleUserText(transcript);
      }
    };

    return recognition;
  };

  const startListening = () => {
    try {
      if (!recognitionRef.current) {
        recognitionRef.current = initRecognition();
      }
      recognitionRef.current?.start();
    } catch (err) {
      console.error(err);
      setError("Could not start microphone. Check permissions.");
    }
  };

  const stopListening = () => {
    recognitionRef.current?.stop();
  };

  const speakText = (text: string) => {
    if (typeof window === "undefined") return;
    const synth = window.speechSynthesis;
    if (!synth) {
      console.warn("speechSynthesis not supported");
      return;
    }
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1;
    utter.pitch = 1;
    synth.speak(utter);
  };

  // ----------- Backend call -----------

  const sendToBackend = async (userText: string) => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/v1/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          user_text: userText,
        }),
      });

      if (!res.ok) {
        const errText = await res.text();
        console.error("API error:", errText);
        throw new Error(`API error: ${res.status}`);
      }

      const data = await res.json();
      const reply = data.reply as string;

      // Add assistant message
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: reply },
      ]);

      // Speak reply
      speakText(reply);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "Unknown error");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Oops, something went wrong talking to the server.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // ----------- Core chat flow -----------

  const handleUserText = (text: string) => {
    if (!text.trim()) return;
    // Add user message to UI
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    // Send to backend
    sendToBackend(text);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed) return;
    setInput("");
    handleUserText(trimmed);
  };

  // Stop speech synthesis when leaving/refreshing
  useEffect(() => {
    return () => {
      if (typeof window !== "undefined") {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex justify-center p-4">
      <div className="w-full max-w-2xl flex flex-col gap-4">
        <h1 className="text-2xl font-semibold text-center mt-4">
          Voice AI Assistant 🎙
        </h1>
        <p className="text-center text-sm text-slate-400">
          STT + TTS in browser • Brain on FastAPI + Supabase + Gemini
        </p>

        {/* Conversation ID (for multi-session later) */}
        <div className="flex gap-2 items-center">
          <label className="text-sm text-slate-300 whitespace-nowrap">
            Conversation ID:
          </label>
          <input
            className="flex-1 rounded-md bg-slate-900 border border-slate-700 px-2 py-1 text-sm"
            value={conversationId}
            onChange={(e) => setConversationId(e.target.value)}
          />
        </div>

        {/* Error */}
        {error && (
          <div className="text-xs text-red-400 bg-slate-900 border border-red-700 rounded-md px-3 py-2">
            {error}
          </div>
        )}

        {/* Chat window */}
        <div className="border border-slate-800 rounded-xl p-3 h-[420px] overflow-y-auto bg-slate-900/70">
          {messages.length === 0 && (
            <p className="text-slate-500 text-sm">
              Start by typing a message below or press the mic button 🎙.
            </p>
          )}

          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`mb-2 flex ${
                m.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`px-3 py-2 rounded-2xl text-sm max-w-[80%] ${
                  m.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-slate-800 text-slate-100"
                }`}
              >
                <div className="font-semibold text-xs mb-1">
                  {m.role === "user" ? "You" : "Assistant"}
                </div>
                <div>{m.content}</div>
              </div>
            </div>
          ))}
          {loading && (
            <p className="text-xs text-slate-400 mt-2">Assistant is thinking…</p>
          )}
        </div>

        {/* Input + buttons */}
        <form onSubmit={handleSubmit} className="flex gap-2 items-center">
          <button
            type="button"
            onClick={isListening ? stopListening : startListening}
            className={`px-3 py-2 rounded-full text-sm font-medium border ${
              isListening
                ? "bg-red-600 border-red-500"
                : "bg-slate-800 border-slate-600"
            }`}
          >
            {isListening ? "Stop 🎙" : "Talk 🎤"}
          </button>

          <input
            className="flex-1 rounded-md bg-slate-900 border border-slate-700 px-3 py-2 text-sm"
            placeholder="Type a message…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />

          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 rounded-md bg-blue-600 hover:bg-blue-500 text-sm font-medium disabled:opacity-50"
          >
            {loading ? "Sending…" : "Send"}
          </button>
        </form>
      </div>
    </main>
  );
}
