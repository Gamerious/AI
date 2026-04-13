import { cn } from "@/lib/utils";

type Level = "prohibited" | "high" | "limited" | "minimal" | "gpai" | "unclassified";

const styles: Record<Level, { label: string; className: string }> = {
  prohibited: {
    label: "Verboten",
    className: "bg-red-50 text-red-700 ring-red-200",
  },
  high: {
    label: "High-Risk",
    className: "bg-orange-50 text-orange-700 ring-orange-200",
  },
  limited: {
    label: "Limited Risk",
    className: "bg-yellow-50 text-yellow-800 ring-yellow-200",
  },
  minimal: {
    label: "Minimal Risk",
    className: "bg-green-50 text-green-700 ring-green-200",
  },
  gpai: {
    label: "GPAI",
    className: "bg-purple-50 text-purple-700 ring-purple-200",
  },
  unclassified: {
    label: "Unklassifiziert",
    className: "bg-gray-100 text-gray-600 ring-gray-200",
  },
};

export function RiskBadge({ level }: { level: Level }) {
  const s = styles[level];
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset",
        s.className,
      )}
    >
      {s.label}
    </span>
  );
}
