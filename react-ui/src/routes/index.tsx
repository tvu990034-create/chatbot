import { useEffect } from "react";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { createThread, ensureHydrated, getState } from "@/lib/console-store";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Local LLM Console - Chat with local models" },
      {
        name: "description",
        content:
          "A text-only console for chatting with local LLM endpoints: model selection, cache control and per-response latency telemetry.",
      },
      { property: "og:title", content: "Local LLM Console" },
      {
        property: "og:description",
        content:
          "A text-only console for chatting with local LLM endpoints: model selection, cache control and per-response latency telemetry.",
      },
    ],
  }),
  component: Index,
});

function Index() {
  const navigate = useNavigate();

  useEffect(() => {
    ensureHydrated();
    const existing = getState().threads[0];
    const threadId = existing ? existing.id : createThread().id;
    void navigate({ to: "/$threadId", params: { threadId }, replace: true });
  }, [navigate]);

  return (
    <div className="flex h-screen items-center justify-center bg-panel">
      <h1 className="num text-[10px] uppercase tracking-widest text-muted-foreground">
        Local LLM Console - loading session
      </h1>
    </div>
  );
}
