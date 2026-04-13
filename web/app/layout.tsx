import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL("https://complai.de"),
  title: {
    default: "ComplAI – EU AI Act Compliance für den Mittelstand",
    template: "%s · ComplAI",
  },
  description:
    "Erfüllen Sie den EU AI Act in Stunden statt Monaten. Self-Service-Software für KI-Inventur, Risikoklassifizierung und auditfähige Dokumentation. Speziell für den deutschsprachigen Mittelstand.",
  keywords: [
    "EU AI Act",
    "KI-Verordnung",
    "AI Act Compliance",
    "KI Compliance",
    "Art. 4 AI Literacy",
    "Risikoklassifizierung KI",
    "DSGVO AI",
    "DACH AI",
  ],
  authors: [{ name: "ComplAI" }],
  openGraph: {
    type: "website",
    locale: "de_DE",
    url: "https://complai.de",
    siteName: "ComplAI",
    title: "ComplAI – Der EU AI Act, erledigt bis Mittag.",
    description:
      "Die erste deutsche Self-Service-Plattform für EU-AI-Act-Compliance. Für Mittelständler, die nicht €50.000 an Big4 zahlen wollen.",
  },
  twitter: {
    card: "summary_large_image",
    title: "ComplAI – EU AI Act erledigt bis Mittag.",
    description:
      "Self-Service-Software für EU-AI-Act-Compliance. Speziell für den DACH-Mittelstand.",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="de" className={inter.variable}>
      <body>{children}</body>
    </html>
  );
}
