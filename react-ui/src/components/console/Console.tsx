import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { checkHealth, fetchModels, sendChat, ApiError } from "@/lib/console-api";
import {
  DEFAULT_MODELS,
  newId,
  titleFromContent,
  type ChatMessage,
} from "@/lib/console-types";

type Health = "unknown" | "checking" | "online" | "offline";

interface ConsoleProps {
  threadId: string;
}

export function Console({ threadId }: ConsoleProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health>("unknown");
  const [apiModels, setApiModels] = useState<string[] | null>(null);
  const [apiUrl, setApiUrl] = useState("http://localhost:8000");
  const [model, setModel] = useState("tinyllama:latest");
  const [useCache, setUseCache] = useState(true);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const availableModels = useMemo(() => {
    const merged = new Set<string>([...DEFAULT_MODELS, ...(apiModels ?? [])]);
    if (model) merged.add(model);
    return [...merged];
  }, [apiModels, model]);

  useEffect(() => {
    textareaRef.current?.focus();
  }, [threadId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, sending]);

  useEffect(() => {
    const controller = new AbortController();
    setHealth("checking");
    const timer = setTimeout(() => {
      void (async () => {
        try {
          await checkHealth(apiUrl, controller.signal);
          setHealth("online");
          try {
            const models = await fetchModels(apiUrl, controller.signal);
            if (models.length) setApiModels(models);
          } catch {
            /* registry is optional */
          }
        } catch {
          if (!controller.signal.aborted) {
            setHealth("offline");
            setApiModels(null);
          }
        }
      })();
    }, 400);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [apiUrl]);

  const send = async () => {
    const content = input.trim();
    if (!content || sending) return;

    setError(null);
    const userMessage: ChatMessage = {
      id: newId(),
      role: "user",
      content,
      createdAt: Date.now(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setSending(true);
    textareaRef.current?.focus();

    try {
      const history = [...messages, userMessage];
      const result = await sendChat(
        { apiUrl, model, useCache },
        history
      );
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: "assistant",
          content: result.response,
          createdAt: Date.now(),
          stats: result.stats,
        },
      ]);
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Unexpected error while calling the API.";
      setError(message);
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: "assistant",
          content: message,
          createdAt: Date.now(),
          failed: true,
        },
      ]);
    } finally {
      setSending(false);
      textareaRef.current?.focus();
    }
  };

  const healthLabel =
    health === "online"
      ? "ENGINE ACTIVE"
      : health === "offline"
        ? "ENGINE UNREACHABLE"
        : health === "checking"
          ? "PROBING ENDPOINT"
          : "STATUS UNKNOWN";

  return (
    <div style={{ display: 'flex', height: '100vh', background: '#111827', color: '#f3f4f6' }}>
      <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #374151', background: '#111827', padding: '12px 16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <h1 style={{ fontSize: '18px', fontWeight: 600 }}>Local Chatbot</h1>
            <div style={{ fontSize: '12px', color: health === "online" ? '#4ade80' : health === "offline" ? '#f87171' : '#fbbf24' }}>
              {healthLabel}
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              style={{ borderRadius: '4px', background: '#1f2937', padding: '4px 12px', fontSize: '14px', color: '#f3f4f6', border: '1px solid #374151' }}
            >
              {availableModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px' }}>
              <input
                type="checkbox"
                checked={useCache}
                onChange={(e) => setUseCache(e.target.checked)}
                style={{ borderRadius: '4px' }}
              />
              Cache
            </label>
          </div>
        </div>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px' }}>
          {messages.length === 0 && (
            <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#6b7280' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ marginBottom: '16px', fontSize: '48px' }}>💬</div>
                <div>Start a conversation</div>
              </div>
            </div>
          )}
          {messages.map((msg) => (
            <div
              key={msg.id}
              style={{
                marginBottom: '16px',
                borderRadius: '8px',
                padding: '12px 16px',
                maxWidth: '80%',
                marginLeft: msg.role === "user" ? 'auto' : 0,
                marginRight: msg.role === "user" ? 0 : 'auto',
                background: msg.role === "user"
                  ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                  : msg.failed
                    ? '#7f1d1d'
                    : '#1f2937',
                color: msg.role === "user" ? 'white' : '#f3f4f6',
              }}
            >
              <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '4px' }}>
                {msg.role === "user" ? "You" : "Assistant"}
                {msg.stats && (
                  <span style={{ marginLeft: '8px' }}>
                    {msg.stats.duration.toFixed(2)}s
                    {msg.stats.cache_hit && " (cached)"}
                    {msg.stats.optimizations_applied && ` (${msg.stats.optimizations_applied} opt)`}
                  </span>
                )}
              </div>
              <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>{msg.content}</div>
            </div>
          ))}
          {sending && (
            <div style={{ marginBottom: '16px', borderRadius: '8px', padding: '12px 16px', background: '#1f2937', color: '#9ca3af' }}>
              Thinking...
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{ borderTop: '1px solid #374151', background: '#111827', padding: '16px' }}>
          {error && (
            <div style={{ marginBottom: '8px', borderRadius: '4px', background: '#7f1d1d', padding: '8px', fontSize: '14px' }}>{error}</div>
          )}
          <div style={{ display: 'flex', gap: '8px' }}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Type your message... (Shift+Enter for newline)"
              style={{
                flex: 1,
                borderRadius: '8px',
                background: '#1f2937',
                padding: '12px',
                fontSize: '14px',
                resize: 'none',
                border: '1px solid #374151',
                color: '#f3f4f6',
                fontFamily: 'inherit',
                minHeight: '50px',
                maxHeight: '150px',
              }}
              rows={2}
              disabled={sending}
            />
            <button
              onClick={send}
              disabled={sending || !input.trim()}
              style={{
                borderRadius: '8px',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                border: 'none',
                padding: '12px 24px',
                fontWeight: 600,
                cursor: sending || !input.trim() ? 'not-allowed' : 'pointer',
                opacity: sending || !input.trim() ? 0.6 : 1,
              }}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
