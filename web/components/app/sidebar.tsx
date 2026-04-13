"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Boxes,
  Compass,
  FileText,
  GraduationCap,
  History,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { name: "AI Inventory", href: "/inventory", icon: Boxes },
  { name: "Risk Classifier", href: "/classify", icon: Compass },
  { name: "Dokumente", href: "/documents", icon: FileText },
  { name: "AI Literacy", href: "/literacy", icon: GraduationCap },
  { name: "Audit Trail", href: "/audit", icon: History },
  { name: "Einstellungen", href: "/settings", icon: Settings },
];

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <aside className="hidden w-60 flex-shrink-0 border-r border-gray-200 bg-white lg:block">
      <nav className="sticky top-16 p-4">
        <ul className="space-y-1">
          {navigation.map((item) => {
            const active = pathname?.startsWith(item.href);
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                    active
                      ? "bg-primary-50 text-primary-700"
                      : "text-gray-700 hover:bg-gray-50",
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  {item.name}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}
