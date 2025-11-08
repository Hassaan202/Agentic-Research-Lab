"use client";

import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowUpRight, Play, UploadCloud, Workflow } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const workflowGlance = [
  "Drop PDFs or arXiv IDs",
  "Agents read & debate",
  "Formatter drafts Collective Insight Report",
];

const quickActions = [
  {
    title: "Upload new corpus",
    description: "PDFs, arXiv IDs, or custom loaders piping into vector storage.",
    href: "/main",
    icon: UploadCloud,
  },
  {
    title: "Configure agent graph",
    description: "Assign LLMs, guardrails, and toolchains per agent role.",
    href: "/main?panel=settings",
    icon: Workflow,
  },
];

export default function HomePage() {
  const router = useRouter();

  return (
    <div className="container space-y-12 py-12">
      <section className="grid gap-10 lg:grid-cols-[1.05fr_0.95fr] lg:items-center">
        <div className="space-y-6">
          <h1 className="text-3xl font-semibold leading-tight sm:text-4xl">
            Operate your research lab from one hub.
          </h1>
          <p className="text-base text-muted-foreground">
            Monitor active agents, upload the next batch of papers, and nudge the workflow forward. Everything is wired for fast backend integration.
          </p>
          <div className="flex flex-wrap gap-4">
            <Button size="lg" onClick={() => router.push("/main")}>Start analysis</Button>
            <Button size="lg" variant="secondary" onClick={() => router.push("/main#uploads")}>
              Upload documents
            </Button>
          </div>
          <Card className="border-white/10 bg-card/80">
            <CardHeader className="space-y-1">
              <CardTitle className="text-lg">Workflow snapshot</CardTitle>
              <CardDescription>Visualise the pipeline before you connect your backend.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              {workflowGlance.map((step, index) => (
                <div key={step} className="flex items-center justify-between rounded-2xl bg-background/60 px-4 py-3">
                  <span className="text-foreground">{step}</span>
                  <span className="text-xs text-muted-foreground">Step {index + 1}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        <Card className="overflow-hidden border-white/10 bg-card/80">
          <div className="relative h-64 w-full">
            <Image
              src="https://images.unsplash.com/photo-1522071820081-009f0129c71c?q=80&w=1600&auto=format&fit=crop"
              alt="Researchers reviewing insights"
              fill
              className="object-cover"
              priority
            />
          </div>
          <CardContent className="space-y-4 p-6">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Play className="h-4 w-4 text-sky-300" />
              Live agent exchange
            </div>
            <p className="text-sm text-muted-foreground">
              Researcher summarises methodology variance, reviewer flags weak evidence, synthesizer drafts the convergence matrix. Everything updates in real time.
            </p>
            <Button asChild variant="secondary">
              <Link href="/main">Open workspace</Link>
            </Button>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-6 md:grid-cols-2">
        {quickActions.map((action) => (
          <Card key={action.title} className="border-white/10 bg-card/80 transition hover:-translate-y-1 hover:border-white/20">
            <CardHeader className="flex flex-row items-start justify-between space-y-0">
              <div className="space-y-2">
                <CardTitle className="text-xl">{action.title}</CardTitle>
                <CardDescription>{action.description}</CardDescription>
              </div>
              <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-white/5">
                <action.icon className="h-5 w-5 text-sky-300" />
              </span>
            </CardHeader>
            <CardContent>
              <Button asChild variant="secondary">
                <Link href={action.href}>Open panel</Link>
              </Button>
            </CardContent>
          </Card>
        ))}
      </section>
    </div>
  );
}
