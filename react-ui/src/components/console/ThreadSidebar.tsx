import { Link } from "@tanstack/react-router";
import type { Thread } from "@/lib/console-types";

type Props = {
  threads: Thread[];
  activeId: string;
  onNewSession: () => void;
  onDelete: (id: string) => void;
  onDeleteAll: () => void;
  onNavigate?: () => void;
};

function formatStamp(ts: number) {
  const d = new Date(ts);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}.${pad(d.getMonth() + 1)}.${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function ThreadSidebar({
  threads,
  activeId,
  onNewSession,
  onDelete,
  onDeleteAll,
  onNavigate,
}: Props) {
  return (
    <div className="flex h-full w-full flex-col border-r border-border bg-panel/50">
      <div className="flex flex-col gap-4 p-4">
        <button
          type="button"
          onClick={onNewSession}
          className="flex w-full items-center justify-between rounded-sm bg-primary px-3 py-2 text-sm font-medium text-primary-foreground ring-1 ring-primary transition-transform active:scale-[0.98]"
        >
          <span>NEW SESSION</span>
          <span className="text-[10px] opacity-50">CTRL+N</span>
        </button>

        <nav className="flex flex-col gap-1">
          <div className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-subtle-foreground">
            Recent Activity
          </div>
          {threads.length === 0 ? (
            <div className="px-3 py-2 text-xs text-subtle-foreground">
              No sessions logged yet.
            </div>
          ) : null}
          {threads.map((thread) => {
            const active = thread.id === activeId;
            return (
              <div
                key={thread.id}
                className={`group flex items-start gap-1 rounded-sm px-1 transition-colors ${
                  active ? "bg-panel-strong/60 ring-1 ring-border-strong" : "hover:bg-panel-strong/30"
                }`}
              >
                <Link
                  to="/$threadId"
                  params={{ threadId: thread.id }}
                  onClick={onNavigate}
                  className="min-w-0 flex-1 px-2 py-2 text-left"
                >
                  <div
                    className={`truncate text-sm ${
                      active ? "font-medium text-foreground" : "text-muted-foreground"
                    }`}
                  >
                    {thread.title}
                  </div>
                  <div className="num mt-0.5 text-[10px] text-subtle-foreground">
                    {formatStamp(thread.updatedAt)} · {thread.messages.length} msg
                  </div>
                </Link>
                <button
                  type="button"
                  aria-label={`Delete session ${thread.title}`}
                  onClick={() => onDelete(thread.id)}
                  className="mt-2 px-2 py-1 text-[10px] font-bold uppercase text-subtle-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100 focus-visible:opacity-100"
                >
                  Del
                </button>
              </div>
            );
          })}
        </nav>
      </div>

      <div className="mt-auto border-t border-border p-4">
        <button
          type="button"
          onClick={onDeleteAll}
          className="num text-[10px] uppercase tracking-wider text-subtle-foreground transition-colors hover:text-destructive"
        >
          Purge all sessions
        </button>
      </div>
    </div>
  );
}
