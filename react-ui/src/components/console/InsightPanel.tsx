import type { ChatMessage } from "@/lib/console-types";

type Props = {
  messages: ChatMessage[];
  availableModels: string[];
  modelsSource: "api" | "default";
};

type ModelRow = {
  model: string;
  calls: number;
  avg: number;
  best: number;
  hits: number;
  optimizations: number;
};

export function buildModelRows(messages: ChatMessage[]): ModelRow[] {
  const map = new Map<string, ModelRow>();
  for (const message of messages) {
    if (!message.stats) continue;
    const { model, duration, cache_hit, optimizations_applied } = message.stats;
    const row =
      map.get(model) ??
      ({ model, calls: 0, avg: 0, best: Number.POSITIVE_INFINITY, hits: 0, optimizations: 0 } as ModelRow);
    row.avg = (row.avg * row.calls + duration) / (row.calls + 1);
    row.calls += 1;
    row.best = Math.min(row.best, duration);
    if (cache_hit) row.hits += 1;
    row.optimizations += optimizations_applied;
    map.set(model, row);
  }
  return [...map.values()].sort((a, b) => a.avg - b.avg);
}

export function InsightPanel({ messages, availableModels, modelsSource }: Props) {
  const rows = buildModelRows(messages);
  const last = [...messages].reverse().find((m) => m.stats)?.stats;
  const slowest = rows.reduce((max, r) => Math.max(max, r.avg), 0);

  return (
    <div className="flex flex-col gap-8 p-6">
      <section>
        <h2 className="mb-4 text-[10px] font-bold uppercase tracking-widest text-subtle-foreground">
          Last Response
        </h2>
        {last ? (
          <div className="space-y-3">
            <div className="flex justify-between text-[11px]">
              <span className="text-muted-foreground">Model</span>
              <span className="num text-foreground">{last.model}</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-muted-foreground">Duration</span>
              <span className="num text-foreground">{last.duration.toFixed(2)}s</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-muted-foreground">Optimizations</span>
              <span className="num text-foreground">{last.optimizations_applied}</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-muted-foreground">Cache</span>
              <span className={`num ${last.cache_hit ? "text-accent" : "text-destructive"}`}>
                {last.cache_hit ? "HIT" : "MISS"}
              </span>
            </div>
            <div className="space-y-1 pt-2">
              <div className="num flex justify-between text-[10px]">
                <span className="text-muted-foreground">OPTIMIZATION LOAD</span>
                <span className="text-foreground">
                  {Math.min(100, last.optimizations_applied)}%
                </span>
              </div>
              <div className="h-1 overflow-hidden rounded-full bg-panel-strong">
                <div
                  className="h-full bg-accent"
                  style={{ width: `${Math.min(100, last.optimizations_applied)}%` }}
                />
              </div>
            </div>
          </div>
        ) : (
          <p className="text-[11px] text-subtle-foreground">
            No responses recorded in this session.
          </p>
        )}
      </section>

      <section>
        <h2 className="mb-4 text-[10px] font-bold uppercase tracking-widest text-subtle-foreground">
          Model Comparison
        </h2>
        {rows.length === 0 ? (
          <p className="text-[11px] text-subtle-foreground">
            Run the same prompt across models to compare latency.
          </p>
        ) : (
          <ul className="space-y-4">
            {rows.map((row) => (
              <li key={row.model} className="space-y-1">
                <div className="num flex justify-between text-[10px]">
                  <span className="text-muted-foreground">{row.model}</span>
                  <span className="text-foreground">{row.avg.toFixed(2)}s avg</span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-panel-strong">
                  <div
                    className="h-full bg-primary"
                    style={{ width: `${slowest ? Math.max(4, (row.avg / slowest) * 100) : 0}%` }}
                  />
                </div>
                <div className="num text-[10px] text-subtle-foreground">
                  {row.calls} calls · best {row.best.toFixed(2)}s · {row.hits} cache hits ·{" "}
                  {row.optimizations} opts
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="rounded-md bg-primary p-4 text-primary-foreground ring-1 ring-primary">
        <h3 className="mb-2 text-[9px] font-bold uppercase text-primary-foreground/50">
          Registry
        </h3>
        <div className="mb-3 text-xs leading-snug text-primary-foreground/80">
          {modelsSource === "api"
            ? "Model list loaded from /api/v1/models."
            : "Using built-in model list. Connect the API to load its registry."}
        </div>
        <ul className="num space-y-1 text-[10px] text-primary-foreground/70">
          {availableModels.map((model) => (
            <li key={model}>{model}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
