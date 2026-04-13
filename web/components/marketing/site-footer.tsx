import Link from "next/link";

export function SiteFooter() {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t border-gray-200 bg-paper py-12">
      <div className="container-narrow">
        <div className="grid gap-8 sm:grid-cols-2 md:grid-cols-4">
          <div>
            <div className="flex items-center gap-2 text-base font-bold tracking-tight">
              <span className="inline-block h-2 w-2 rounded-full bg-primary" aria-hidden />
              Compl<span className="text-primary">AI</span>
            </div>
            <p className="mt-3 text-xs text-gray-500">
              EU AI Act Compliance.
              <br />
              Hosted in Frankfurt. Built in DACH.
            </p>
          </div>

          <FooterColumn
            title="Produkt"
            links={[
              { label: "Module", href: "#module" },
              { label: "Preise", href: "#preise" },
              { label: "FAQ", href: "#faq" },
              { label: "Roadmap", href: "/roadmap" },
            ]}
          />

          <FooterColumn
            title="Wissen"
            links={[
              { label: "Blog", href: "/blog" },
              { label: "AI-Act-Wiki", href: "/wiki" },
              { label: "Risiko-Check", href: "/risiko-check" },
              { label: "Webinare", href: "/webinare" },
            ]}
          />

          <FooterColumn
            title="Rechtliches"
            links={[
              { label: "Impressum", href: "/impressum" },
              { label: "Datenschutz", href: "/datenschutz" },
              { label: "AGB", href: "/agb" },
              { label: "AVV", href: "/avv" },
              { label: "Sub-Processors", href: "/legal/subprocessors" },
            ]}
          />
        </div>

        <div className="mt-10 flex flex-col items-start justify-between gap-4 border-t border-gray-200 pt-6 text-xs text-gray-500 md:flex-row md:items-center">
          <p>© {year} ComplAI. Alle Rechte vorbehalten.</p>
          <p>
            Hinweis: ComplAI ist eine Software-Lösung, keine Rechtsberatung. Für rechtsverbindliche
            Auskünfte konsultieren Sie bitte eine spezialisierte Kanzlei.
          </p>
        </div>
      </div>
    </footer>
  );
}

function FooterColumn({
  title,
  links,
}: {
  title: string;
  links: { label: string; href: string }[];
}) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      <ul className="mt-3 space-y-2">
        {links.map((link) => (
          <li key={link.href}>
            <Link href={link.href} className="text-xs text-gray-600 hover:text-primary">
              {link.label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
