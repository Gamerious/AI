import { WaitlistForm } from "./waitlist-form";

export function WaitlistSection() {
  return (
    <section id="waitlist" className="bg-ink py-20 text-white">
      <div className="container-narrow">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
            Sichern Sie sich Early Access.
          </h2>
          <p className="mt-4 text-lg text-gray-300">
            Wir launchen ComplAI im Mai 2026. Die ersten 50 Unternehmen auf der Warteliste bekommen
            <span className="font-semibold text-white"> 50 % Rabatt für 6 Monate</span> und
            persönliches Onboarding.
          </p>
        </div>

        <div className="mx-auto mt-10 max-w-md">
          <WaitlistForm />
        </div>

        <p className="mt-6 text-center text-xs text-gray-400">
          Wir senden Ihnen nur Updates zu unserem Launch. Keine Werbe-Mails.
          Datenverarbeitung gemäß DSGVO. Jederzeit abbestellbar.
        </p>
      </div>
    </section>
  );
}
