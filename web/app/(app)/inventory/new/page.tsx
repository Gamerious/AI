import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { NewSystemForm } from "@/components/app/new-system-form";

export default function NewSystemPage() {
  return (
    <div className="mx-auto max-w-2xl">
      <Link
        href="/inventory"
        className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-ink"
      >
        <ArrowLeft className="h-4 w-4" />
        Zur Inventur
      </Link>

      <h1 className="mt-4 text-3xl font-bold text-ink">Neues KI-System anlegen</h1>
      <p className="mt-2 text-sm text-gray-600">
        Tragen Sie ein KI-System in Ihre Inventur ein. Sie können es später jederzeit ergänzen und
        klassifizieren.
      </p>

      <div className="mt-8 rounded-2xl border border-gray-200 bg-white p-6">
        <NewSystemForm />
      </div>
    </div>
  );
}
