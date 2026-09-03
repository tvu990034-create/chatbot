import { createFileRoute } from "@tanstack/react-router";
import { Console } from "@/components/console/Console";

export const Route = createFileRoute("/$threadId")({
  head: () => ({
    meta: [
      { title: "Session - Local LLM Console" },
      {
        name: "description",
        content:
          "Chat session against a local LLM endpoint with per-response latency, optimization count and cache telemetry.",
      },
      { property: "og:title", content: "Session - Local LLM Console" },
      {
        property: "og:description",
        content:
          "Chat session against a local LLM endpoint with per-response latency, optimization count and cache telemetry.",
      },
    ],
  }),
  component: ThreadPage,
});

function ThreadPage() {
  const { threadId } = Route.useParams();
  return <Console key={threadId} threadId={threadId} />;
}
