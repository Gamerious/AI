import Link from "next/link";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-gray-200 bg-paper/80 backdrop-blur">
      <div className="container-narrow flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-2 text-lg font-bold tracking-tight">
          <span className="inline-block h-2 w-2 rounded-full bg-primary" aria-hidden />
          Compl<span className="text-primary">AI</span>
        </Link>
        <nav className="hidden items-center gap-8 md:flex">
          <Link href="#problem" className="text-sm text-gray-700 hover:text-ink">
            Problem
          </Link>
          <Link href="#module" className="text-sm text-gray-700 hover:text-ink">
            Module
          </Link>
          <Link href="#preise" className="text-sm text-gray-700 hover:text-ink">
            Preise
          </Link>
          <Link href="#faq" className="text-sm text-gray-700 hover:text-ink">
            FAQ
          </Link>
        </nav>
        <Link href="#waitlist" className="button-primary text-xs">
          Early Access
        </Link>
      </div>
    </header>
  );
}
