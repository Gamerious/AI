import type { Metadata } from "next";
import { AppSidebar } from "@/components/app/sidebar";
import { AppHeader } from "@/components/app/header";

export const metadata: Metadata = {
  title: "App · ComplAI",
  description: "Ihre EU-AI-Act-Compliance-Arbeitsumgebung.",
  robots: { index: false, follow: false },
};

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-paper">
      <AppHeader />
      <div className="flex">
        <AppSidebar />
        <main className="flex-1 p-6 lg:p-8">{children}</main>
      </div>
    </div>
  );
}
