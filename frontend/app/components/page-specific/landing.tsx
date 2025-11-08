"use client";

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useState } from "react";
import {
  ArrowRight,
  BrainCircuit,
  FileBarChart,
  MessageSquare,
  Network,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

import type { LucideIcon } from "lucide-react";

const stats = [
  { value: "4.3x", label: "Faster synthesis" },
  { value: "92%", label: "Citations traced" },
  { value: "< 8 min", label: "Insight turnaround" },
];

const featureHighlights: Array<{
  title: string;
  description: string;
  icon: LucideIcon;
}> = [
  {
    title: "Grounded ingestion",
    description: "Drop in PDFs or arXiv links and extract claims, metrics, and entities in one sweep.",
    icon: BrainCircuit,
  },
  {
    title: "Critical dialogue",
    description: "Reviewer and questioner agents debate findings before they ever hit your report.",
    icon: MessageSquare,
  },
  {
    title: "Formatter-ready output",
    description: "Ship Collective Insight Reports with citations and follow-up hypotheses instantly.",
    icon: FileBarChart,
  },
];

const flowSteps = [
  "Upload documents",
  "Researcher distills evidence",
  "Reviewer stress-tests claims",
  "Synthesizer maps insights",
  "Formatter outputs report",
];

const exampleReport = `Collective Insight Report — Renewable Energy Storage (Sample)\n\nClaim strength: 78% of reviewed papers corroborate AI-augmented predictive maintenance reducing downtime by 18-26%.\nDivergence: Two recent preprints highlight data drift when turbine telemetry is not domain-adapted.\nHypothesis: Active transfer learning over multimodal sensor streams could stabilise forecasting horizons beyond 72 hours.\n\nCitations: [Nguyen 2024], [Bosch 2025], [NREL Dataset 2025].`;

export default function LandingPage() {
  const router = useRouter();
  const [previewOpen, setPreviewOpen] = useState(false);

  return (
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0 -z-10 bg-hero-grid opacity-70" />
      <div className="absolute -left-40 top-20 h-72 w-72 rounded-full bg-sky-500/30 blur-3xl" />
      <div className="absolute -right-32 bottom-0 h-72 w-72 rounded-full bg-purple-500/20 blur-3xl" />

      <section className="relative py-20 sm:py-28">
        <div className="container grid gap-14 lg:grid-cols-[1.05fr_0.95fr] lg:gap-24">
          <div className="space-y-8">
            <span className="section-heading animate-fade-in-up">VC Big Bets · Research Track</span>
            <h1 className="animate-fade-in-up text-4xl font-semibold leading-tight tracking-tight text-foreground sm:text-5xl lg:text-6xl">
              A mini research lab where agents think together.
            </h1>
            <p className="animate-fade-in-up text-lg text-muted-foreground sm:text-xl">
              Researcher, reviewer, synthesizer, questioner, formatter—each agent plays its part so you can uncover, defend, and ship new hypotheses faster than a single prompt ever could.
            </p>

            <div className="flex flex-wrap items-center gap-3 animate-fade-in-up">
              <Button size="lg" onClick={() => router.push("/home")}>
                Launch workspace
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
              <Button
                size="lg"
                variant="secondary"
                onClick={() => router.push("/main")}
                className="border border-white/10 bg-transparent text-foreground backdrop-blur"
              >
                Try the sandbox
              </Button>
              <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
                <DialogTrigger asChild>
                  <Button variant="ghost" className="text-sm text-muted-foreground">
                    View sample report
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-2xl border border-white/10 bg-background/95">
                  <DialogHeader>
                    <DialogTitle>Collective Insight Report</DialogTitle>
                    <DialogDescription>
                      Formatter agent output—swap this text with your backend payload.
                    </DialogDescription>
                  </DialogHeader>
                  <pre className="max-h-[420px] overflow-y-auto rounded-lg bg-card/80 p-6 text-sm leading-relaxed text-muted-foreground shadow-inner">
{exampleReport}
                  </pre>
                </DialogContent>
              </Dialog>
            </div>

            <dl className="grid grid-cols-3 gap-6 pt-10">
              {stats.map((item) => (
                <div key={item.label} className="space-y-2">
                  <dt className="text-xs uppercase tracking-widest text-muted-foreground">
                    {item.label}
                  </dt>
                  <dd className="text-3xl font-semibold text-foreground">{item.value}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="relative">
            <div className="glass-panel animate-float rounded-3xl border-white/10 bg-white/5 p-6 shadow-[0_0_120px_rgb(59_130_246_/_0.25)]">
              <div className="relative aspect-[5/4] w-full overflow-hidden rounded-2xl border border-white/10">
                <Image
                  src="https://images.unsplash.com/photo-1520607162513-77705c0f0d4a?q=80&w=1400&auto=format&fit=crop"
                  alt="Researchers collaborating in a lab"
                  fill
                  className="object-cover object-center"
                  priority
                />
              </div>
              <div className="mt-6 rounded-2xl border border-white/10 bg-background/80 p-5 shadow-md">
                <div className="flex items-center justify-between text-xs uppercase tracking-[0.3em] text-primary/70">
                  <span>Live multi-agent loop</span>
                  <span>Streaming</span>
                </div>
                <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground">
                  <span>Researcher → Reviewer → Synthesizer</span>
                  <span className="text-foreground">v0.9</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="overview" className="border-t border-white/5 py-20">
        <div className="container grid gap-12 lg:grid-cols-[1fr_0.9fr] lg:items-center">
          <div className="space-y-5">
            <span className="section-heading">What you get</span>
            <h2 className="text-3xl font-semibold sm:text-4xl">
              Built to plug into your backend, styled for investor-ready demos.
            </h2>
            <p className="text-base text-muted-foreground">
              Every surface is powered by shadcn/ui components, so you can wire your own API responses, stream events, or swap themes without fighting the layout.
            </p>

            <div className="grid gap-4 sm:grid-cols-2">
              {featureHighlights.map((feature) => (
                <Card key={feature.title} className="border-white/10 bg-card/80">
                  <CardHeader className="space-y-2">
                    <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/5 text-white">
                      <feature.icon className="h-5 w-5" />
                    </span>
                    <CardTitle className="text-lg">{feature.title}</CardTitle>
                    <CardDescription className="text-sm leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </CardHeader>
                </Card>
              ))}
            </div>
          </div>

          <Card className="border-white/10 bg-card/80">
            <CardHeader>
              <CardTitle>Wire it up in minutes</CardTitle>
              <CardDescription>Drop your own agent events into this UI shell.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm text-muted-foreground">
              <pre className="rounded-2xl border border-white/10 bg-background/70 p-5 font-mono text-xs leading-relaxed">
{`{
  "agent": "Reviewer",
  "status": "completed",
  "objections": ["insufficient sample size", "missing citations"],
  "next": "Synthesizer"
}`}
              </pre>
              <p>
                Send events via REST, SSE, or MCP—components already expose loaders, progress bars, and placeholders for verification data.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      <section className="border-t border-white/5 bg-background/70 py-20">
        <div className="container grid gap-12 lg:grid-cols-[0.85fr_1.15fr] lg:items-center">
          <div className="space-y-6">
            <span className="section-heading">Workflow glimpse</span>
            <h2 className="text-3xl font-semibold sm:text-4xl">
              Visualise the reasoning chain before you wire the backend.
            </h2>
            <p className="text-base text-muted-foreground">
              The interface guides operators through every stage, so the PDF upload, agent chatter, and final report all live in one coherent flow.
            </p>
            <div className="rounded-3xl border border-white/10 bg-card/80 p-6">
              <div className="flex items-center gap-3 text-sm font-medium text-foreground">
                <Network className="h-5 w-5 text-sky-300" />
                Orchestrated pipeline (sample)
              </div>
              <div className="mt-5 grid gap-3 text-sm text-muted-foreground">
                {flowSteps.map((step, index) => (
                  <div key={step} className="flex items-center justify-between rounded-xl bg-background/60 px-4 py-3">
                    <span className="text-foreground">{step}</span>
                    <span className="text-xs text-muted-foreground">Step {index + 1}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <Card className="border-white/10 bg-card/80">
            <CardHeader>
              <CardTitle>Formatter snapshot</CardTitle>
              <CardDescription>Swap this placeholder for your live insight payload.</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="max-h-[360px] overflow-y-auto rounded-2xl border border-white/10 bg-background/70 p-6 font-mono text-xs leading-relaxed text-muted-foreground">
{exampleReport}
              </pre>
            </CardContent>
          </Card>
        </div>
      </section>

      <section className="border-t border-white/5 py-20">
        <div className="container flex flex-col items-center gap-6 text-center">
          <span className="section-heading">Ready to launch?</span>
          <h2 className="max-w-3xl text-3xl font-semibold sm:text-4xl">
            Deploy the UI, connect your agents, and show the world how collaborative reasoning feels.
          </h2>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <Button
              size="lg"
              onClick={() => router.push("/home")}
              className="shadow-[0_0_45px_rgb(14_165_233_/_0.35)]"
            >
              Explore the home hub
            </Button>
            <Button
              size="lg"
              variant="secondary"
              onClick={() => router.push("/main")}
              className="border border-white/10"
            >
              Open the workspace
            </Button>
          </div>
        </div>
      </section>
    </div>
  );
}
