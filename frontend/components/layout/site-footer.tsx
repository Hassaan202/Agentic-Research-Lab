import Link from "next/link";
import { Github, Linkedin, Mail } from "lucide-react";

const year = new Date().getFullYear();

export function SiteFooter() {
  return (
    <footer className="border-t border-white/5 bg-background/80">
      <div className="container flex flex-col gap-8 py-12 md:flex-row md:items-center md:justify-between">
        <div className="space-y-3">
          <div className="text-sm font-semibold uppercase tracking-[0.4em] text-primary/60">
            Agentic Research Lab
          </div>
          <p className="max-w-md text-sm text-muted-foreground">
            Prototype a trustworthy multi-agent research assistant. Built with a modular frontend so you can plug in your own data sources, vector stores, and reasoning engines.
          </p>
        </div>
        <div className="flex flex-col gap-4 text-sm">
          <div className="flex items-center gap-3 text-muted-foreground">
            <Mail className="h-4 w-4" />
            <Link href="mailto:team@agentic.ai" className="transition hover:text-foreground">
              team@agentic.ai
            </Link>
          </div>
          <div className="flex items-center gap-3 text-muted-foreground">
            <Github className="h-4 w-4" />
            <Link href="https://github.com" target="_blank" rel="noreferrer" className="transition hover:text-foreground">
              GitHub
            </Link>
          </div>
          <div className="flex items-center gap-3 text-muted-foreground">
            <Linkedin className="h-4 w-4" />
            <Link href="https://www.linkedin.com" target="_blank" rel="noreferrer" className="transition hover:text-foreground">
              LinkedIn
            </Link>
          </div>
        </div>
      </div>
      <div className="border-t border-white/5 py-6">
        <div className="container flex flex-col justify-between gap-2 text-xs text-muted-foreground md:flex-row">
          <p>© {year} Research Agent. Crafted in the VC Big Bets track.</p>
          <div className="flex gap-4">
            <Link href="/privacy" className="transition hover:text-foreground">
              Privacy
            </Link>
            <Link href="/terms" className="transition hover:text-foreground">
              Terms
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
