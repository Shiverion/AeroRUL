import type { RiskTier } from "../api";

const LABELS: Record<RiskTier, string> = {
  low: "Low",
  medium: "Medium",
  high: "High",
  critical: "Critical",
};

export function RiskBadge({ tier }: { tier: RiskTier }) {
  return <span className={`risk-badge risk-${tier}`}>{LABELS[tier]}</span>;
}
