import { useEffect, useState } from "react";

/**
 * Where the interface is, kept in the URL hash.
 *
 * A screening queue is worked over hours by more than one person, so a case
 * needs to survive a refresh and be sendable to a colleague. The hash is
 * enough for that and needs no router: three tabs and an optional case id.
 *
 *   #/screen              #/queue              #/queue/<screening id>
 */
export type Tab = "screen" | "queue" | "sim";

export interface Route {
  tab: Tab;
  caseId: string | null;
}

const TABS: Tab[] = ["screen", "queue", "sim"];

export function parseHash(hash: string): Route {
  const [, tab, caseId] = hash.replace(/^#\/?/, "/").split("/");
  return {
    tab: TABS.includes(tab as Tab) ? (tab as Tab) : "screen",
    caseId: caseId || null,
  };
}

export function formatHash({ tab, caseId }: Route): string {
  return caseId && tab === "queue" ? `#/queue/${caseId}` : `#/${tab}`;
}

export function useRoute(): [Route, (next: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));

  useEffect(() => {
    const onHashChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  return [
    route,
    (next: Route) => {
      window.location.hash = formatHash(next);
      setRoute(next);
    },
  ];
}
