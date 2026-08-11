import type { ReactNode } from "react";

export function InfoTip({ children }: { children: ReactNode }) {
  return (
    <span className="info-tip" tabIndex={0}>
      i
      <span className="tip-bubble">{children}</span>
    </span>
  );
}
