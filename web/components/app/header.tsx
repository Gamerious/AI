import Link from "next/link";
import { Bell, User } from "lucide-react";

export function AppHeader() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-gray-200 bg-white">
      <div className="flex h-16 items-center justify-between px-4 lg:px-6">
        <Link href="/dashboard" className="flex items-center gap-2 text-lg font-bold tracking-tight">
          <span className="inline-block h-2 w-2 rounded-full bg-primary" aria-hidden />
          Compl<span className="text-primary">AI</span>
        </Link>

        <div className="flex items-center gap-3">
          <button
            type="button"
            className="rounded-full p-2 text-gray-500 transition-colors hover:bg-gray-100 hover:text-ink"
            aria-label="Benachrichtigungen"
          >
            <Bell className="h-5 w-5" />
          </button>
          <button
            type="button"
            className="flex items-center gap-2 rounded-full border border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-ink hover:bg-gray-50"
          >
            <User className="h-4 w-4" />
            <span className="hidden sm:inline">Demo-Account</span>
          </button>
        </div>
      </div>
    </header>
  );
}
