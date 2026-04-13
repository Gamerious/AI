import { Hero } from "@/components/marketing/hero";
import { ProblemSection } from "@/components/marketing/problem-section";
import { SolutionSection } from "@/components/marketing/solution-section";
import { ModulesGrid } from "@/components/marketing/modules-grid";
import { PricingSection } from "@/components/marketing/pricing-section";
import { FAQSection } from "@/components/marketing/faq-section";
import { WaitlistSection } from "@/components/marketing/waitlist-section";
import { SiteHeader } from "@/components/marketing/site-header";
import { SiteFooter } from "@/components/marketing/site-footer";

export default function HomePage() {
  return (
    <>
      <SiteHeader />
      <main>
        <Hero />
        <ProblemSection />
        <SolutionSection />
        <ModulesGrid />
        <PricingSection />
        <FAQSection />
        <WaitlistSection />
      </main>
      <SiteFooter />
    </>
  );
}
