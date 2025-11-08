"use client";

import { ChangeEvent, useMemo, useState } from "react";
import {
  ArrowRightCircle,
  BrainCircuit,
  MessageSquare,
  ShieldCheck,
  Sparkles,
  Workflow,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";

type AgentStatus = "pending" | "working" | "completed";

type AgentStage = {
  name: string;
  role: string;
  signal: string;
  icon: typeof BrainCircuit;
};

const AGENT_PIPELINE: AgentStage[] = [
  {
    name: "Researcher",
    role: "Ingests docs, extracts entities, normalises metrics",
    signal: "Structured source notebook",
    icon: BrainCircuit,
  },
  {
    name: "Reviewer",
    role: "Red teams claims, flags weak evidence, requests clarifications",
    signal: "Objection + risk ledger",
    icon: ShieldCheck,
  },
  {
    name: "Synthesizer",
    role: "Maps convergences, divergences, and opportunity areas",
    signal: "Insight graph + hypothesis candidates",
    icon: Sparkles,
  },
  {
    name: "Questioner",
    role: "Surfaces gaps, missing datasets, follow-up prompts",
    signal: "Gap checklist",
    icon: MessageSquare,
  },
  {
    name: "Formatter",
    role: "Packages the Collective Insight Report with citations",
    signal: "PDF + JSON payload",
    icon: Workflow,
  },
];

export default function MainPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [agentStates, setAgentStates] = useState<Array<AgentStage & { status: AgentStatus }>>(
    () => AGENT_PIPELINE.map((agent) => ({ ...agent, status: "pending" }))
  );
  const [logs, setLogs] = useState<string[]>([]);
  const [report, setReport] = useState("");

  const pipelineReady = useMemo(() => files.length > 0 && !isAnalyzing, [files.length, isAnalyzing]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) return;
    setFiles(Array.from(event.target.files));
  };

  const resetPipeline = () => {
    setAgentStates(AGENT_PIPELINE.map((stage) => ({ ...stage, status: "pending" })));
    setAnalysisProgress(0);
    setLogs([]);
    setReport("");
  };

  const startAnalysis = async () => {
    if (!files.length) return;

    setIsAnalyzing(true);
    setReport("");
    setLogs([]);
    setAgentStates(AGENT_PIPELINE.map((stage, index) => ({ ...stage, status: index === 0 ? "working" : "pending" })));

    for (let index = 0; index < AGENT_PIPELINE.length; index++) {
      const stage = AGENT_PIPELINE[index];
      setAgentStates((prev) =>
        prev.map((agent, agentIndex) => {
          if (agentIndex < index) {
            return { ...agent, status: "completed" };
          }
          if (agentIndex === index) {
            return { ...agent, status: "working" };
          }
          return { ...agent, status: "pending" };
        })
      );

      setLogs((prev) => [
        ...prev,
        `${stage.name}: ${stage.role} → emits ${stage.signal}`,
      ]);

      await new Promise((resolve) => setTimeout(resolve, 1400));

      setAgentStates((prev) =>
        prev.map((agent, agentIndex) =>
          agentIndex === index ? { ...agent, status: "completed" } : agent
        )
      );

      setAnalysisProgress(((index + 1) / AGENT_PIPELINE.length) * 100);
    }

    setReport(`Collective Insight Report (sample output)\n\n• Claim: Predictive maintenance agents reduce unexpected downtime by 18-26% across evaluated papers.\n• Divergence: Domain adaptation is required when telemetry drifts beyond 12 weeks.\n• Hypothesis: Active transfer learning on multimodal sensor streams can extend forecasting stability to 72h+.\n\nNext steps: validate on historical telemetry + connect reviewer feedback loop.`);
    setIsAnalyzing(false);
  };

  return (
    <div className="container mx-auto flex max-w-5xl flex-col gap-10 py-12" id="uploads">
      <section className="space-y-3 text-center">
        <p className="text-xs uppercase tracking-[0.3em] text-primary/70">Workspace</p>
        <h2 className="text-3xl font-semibold text-foreground sm:text-4xl">Central intelligence console</h2>
        <p className="mx-auto max-w-2xl text-sm text-muted-foreground">
          Bring source material into the reasoning loop, monitor every agent hand-off, and publish collective
          insights in a single, distraction-free lane.
        </p>
      </section>

      <Card className="border-white/10 bg-card/80 backdrop-blur">
        <CardHeader className="space-y-3">
          <CardTitle>Knowledge ingestion</CardTitle>
          <CardDescription>Upload PDFs, notes, or connect your custom loader.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="rounded-3xl border-2 border-dashed border-white/10 bg-background/60 p-6 text-center">
            <Label htmlFor="research-files" className="flex flex-col items-center gap-3 text-muted-foreground">
              <ArrowRightCircle className="h-8 w-8 text-sky-300" />
              <span className="text-sm">Drag & drop or click to select up to 10 PDF files</span>
            </Label>
            <Input
              id="research-files"
              type="file"
              accept="application/pdf"
              multiple
              className="hidden"
              onChange={handleFileChange}
            />
          </div>

          {files.length > 0 && (
            <div className="rounded-2xl border border-white/10 bg-background/70 p-4">
              <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">Queued documents</p>
              <Separator className="my-3 bg-white/10" />
              <ul className="space-y-2 text-sm">
                {files.map((file) => (
                  <li key={file.name + file.lastModified} className="flex items-center justify-between text-muted-foreground">
                    <span className="truncate pr-4 text-foreground">{file.name}</span>
                    <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </CardContent>
        <CardFooter className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <Button
            variant="ghost"
            onClick={resetPipeline}
            disabled={files.length === 0 && !report}
          >
            Reset workspace
          </Button>
          <Button onClick={startAnalysis} disabled={!pipelineReady} className="w-full sm:w-auto">
            {isAnalyzing ? "Analyzing..." : "Start analysis"}
          </Button>
        </CardFooter>
      </Card>

      <Card className="border-white/10 bg-card/80 backdrop-blur">
        <CardHeader className="space-y-3">
          <CardTitle>Agent pipeline</CardTitle>
          <CardDescription>Live status across the orchestrated reasoning chain.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3 rounded-2xl border border-white/10 bg-background/60 p-4">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-xs uppercase tracking-[0.3em] text-muted-foreground">Pipeline progress</div>
              <div className="text-sm text-foreground">{Math.round(analysisProgress)}% complete</div>
            </div>
            <Progress value={analysisProgress} className="h-2" />
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {agentStates.map((agent) => (
              <div
                key={agent.name}
                className="rounded-2xl border border-white/10 bg-background/60 p-4"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/5">
                      <agent.icon className="h-5 w-5 text-sky-300" />
                    </span>
                    <div>
                      <p className="text-sm font-semibold text-foreground">{agent.name}</p>
                      <p className="text-xs text-muted-foreground">{agent.signal}</p>
                    </div>
                  </div>
                  <span
                    className={
                      agent.status === "completed"
                        ? "text-xs font-medium text-emerald-400"
                        : agent.status === "working"
                        ? "text-xs font-medium text-yellow-300"
                        : "text-xs text-muted-foreground"
                    }
                  >
                    {agent.status}
                  </span>
                </div>
                <p className="mt-3 text-xs text-muted-foreground">{agent.role}</p>
              </div>
            ))}
          </div>
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="ghost" className="justify-start px-0 text-sm text-muted-foreground">
                View agent message trace
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-xl border border-white/10 bg-background/95">
              <DialogHeader>
                <DialogTitle>Agent log (synthetic)</DialogTitle>
              </DialogHeader>
              <div className="max-h-[360px] space-y-3 overflow-y-auto rounded-lg bg-card/70 p-4 text-sm text-muted-foreground">
                {logs.length === 0 ? (
                  <p>No logs yet — start an analysis to populate this feed.</p>
                ) : (
                  logs.map((log, index) => (
                    <div key={index} className="flex gap-2">
                      <span className="text-xs text-muted-foreground">{index + 1}.</span>
                      <p>{log}</p>
                    </div>
                  ))
                )}
              </div>
            </DialogContent>
          </Dialog>
        </CardContent>
      </Card>

      <Card className="border-white/10 bg-card/80 backdrop-blur">
        <CardHeader>
          <CardTitle>Collective insight report</CardTitle>
          <CardDescription>Formatter-ready payload. Swap with real backend response.</CardDescription>
        </CardHeader>
        <CardContent>
          <Textarea
            value={report}
            onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setReport(event.target.value)}
            rows={12}
            placeholder="Report will populate once the pipeline completes."
          />
        </CardContent>
        <CardFooter className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <span className="text-xs text-muted-foreground">Edit before exporting or wire to your transport.</span>
          <Button variant="secondary" disabled={!report}>
            Export JSON
          </Button>
        </CardFooter>
      </Card>

      <Card className="border-white/10 bg-card/80 backdrop-blur">
        <CardHeader>
          <CardTitle>Integration checklist</CardTitle>
          <CardDescription>Hook the UI into your orchestration backend.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 text-sm text-muted-foreground sm:grid-cols-2">
          <div className="rounded-2xl border border-white/10 bg-background/60 p-4">
            <p className="text-xs uppercase tracking-[0.3em] text-primary/70">REST hooks</p>
            <p className="mt-2">
              POST <span className="font-mono text-xs text-foreground">/api/analyze</span> with uploaded file references. Stream SSE
              events to update the status feed in real time.
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-background/60 p-4">
            <p className="text-xs uppercase tracking-[0.3em] text-primary/70">MCP ready</p>
            <p className="mt-2">
              Implement the Model Context Protocol (MCP) interface to let external agents subscribe to reasoning updates right from
              this UI.
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-background/60 p-4">
            <p className="text-xs uppercase tracking-[0.3em] text-primary/70">Observability</p>
            <p className="mt-2">
              Stream <span className="font-mono text-xs text-foreground">agent_state</span> events into your analytics warehouse and power dashboards or alerting.
            </p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-background/60 p-4">
            <p className="text-xs uppercase tracking-[0.3em] text-primary/70">Next actions</p>
            <p className="mt-2">
              Map the report payload to your preferred schema, then trigger human-in-the-loop reviews before publishing decisions.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
