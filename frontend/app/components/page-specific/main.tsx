"use client";

import { ChangeEvent, useMemo, useState, useRef, useEffect } from "react";
import {
  ArrowRightCircle,
  BrainCircuit,
  ChevronDown,
  ChevronUp,
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
import {
  startResearch,
  pollResearchStatus,
  getCollaborationLog,
  getResearchStatus,
  getResearchResult,
  uploadDocuments,
  processDocuments,
  getProcessStatus,
  listDocuments,
  deleteDocument,
  type ResearchStatus,
  type ResearchResult,
  type DocumentInfo,
} from "@/lib/api";

type AgentStatus = "pending" | "working" | "completed" | "error";

type AgentStage = {
  name: string;
  role: string;
  signal: string;
  icon: typeof BrainCircuit;
  statusMessage?: string;
  detailedOutput?: {
    analysis?: string;
    findings?: Array<any>;
    critique?: string;
    strengths?: string[];
    weaknesses?: string[];
    synthesis?: string;
    hypotheses?: Array<any>;
    gapAnalysis?: string;
    questions?: string[];
    sources?: Array<any>;
  };
};

const AGENT_PIPELINE: AgentStage[] = [
  {
    name: "Researcher",
    role: "Analyzes research papers and extracts key findings",
    signal: "Key findings with citations",
    icon: BrainCircuit,
  },
  {
    name: "Reviewer",
    role: "Critiques findings and identifies strengths/weaknesses",
    signal: "Critical analysis and evaluation",
    icon: ShieldCheck,
  },
  {
    name: "Synthesizer",
    role: "Synthesizes insights and generates testable hypotheses",
    signal: "Novel hypotheses and insights",
    icon: Sparkles,
  },
  {
    name: "Questioner",
    role: "Identifies research gaps and generates follow-up questions",
    signal: "Research questions and gaps",
    icon: MessageSquare,
  },
  {
    name: "Formatter",
    role: "Compiles comprehensive research report with citations",
    signal: "Formatted research report",
    icon: Workflow,
  },
];

export default function MainPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploadedDocuments, setUploadedDocuments] = useState<DocumentInfo[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<string | null>(null);
  const [processId, setProcessId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [agentStates, setAgentStates] = useState<Array<AgentStage & { status: AgentStatus }>>(
    () => AGENT_PIPELINE.map((agent) => ({ ...agent, status: "pending" }))
  );
  const [logs, setLogs] = useState<string[]>([]);
  const [report, setReport] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null);
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const processingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const pipelineReady = useMemo(() => !isAnalyzing, [isAnalyzing]);

  // Map backend agent names to frontend agent names
  const agentNameMap: Record<string, string> = {
    RESEARCHER: "Researcher",
    REVIEWER: "Reviewer",
    SYNTHESIZER: "Synthesizer",
    QUESTIONER: "Questioner",
    FORMATTER: "Formatter",
  };

  // Update agent states based on current step
  // Backend current_step: 0=initial, 1=researcher, 2=reviewer, 3=synthesizer, 4=questioner, 5=formatter
  // Frontend agents: 0=researcher, 1=reviewer, 2=synthesizer, 3=questioner, 4=formatter
  const updateAgentStatesFromStep = (currentStep: number, workflowStatus: string) => {
    setAgentStates((prev) =>
      prev.map((agent, index) => {
        if (workflowStatus === "error") {
          return { ...agent, status: "error" };
        }
        // currentStep 1 = researcher (index 0), currentStep 2 = reviewer (index 1), etc.
        const agentStep = index + 1;
        if (agentStep < currentStep) {
          return { ...agent, status: "completed" };
        }
        if (agentStep === currentStep && (workflowStatus === "running" || workflowStatus === "started")) {
          return { ...agent, status: "working" };
        }
        if (workflowStatus === "success") {
          return { ...agent, status: "completed" };
        }
        return { ...agent, status: "pending" };
      })
    );

    // Update progress based on current step (0-5 steps = 0-100%)
    // Step 5 (formatter) = 100%, Step 0 = 0%
    const progress = Math.min((currentStep / 5) * 100, 100);
    setAnalysisProgress(progress);
  };

  // Update logs from collaboration log
  const updateLogsFromCollaboration = (collaborationLog: ResearchResult["collaboration_log"]) => {
    if (!collaborationLog) return;

    const logEntries = collaborationLog.map((entry) => {
      const agentName = agentNameMap[entry.agent] || entry.agent;
      const actionDesc = entry.action.replace(/_/g, " ");
      const outputPreview = entry.output 
        ? Object.entries(entry.output)
            .slice(0, 2)
            .map(([key, val]) => {
              if (typeof val === "string") return `${key}: ${val.substring(0, 50)}...`;
              if (Array.isArray(val)) return `${key}: ${val.length} items`;
              return `${key}: ${JSON.stringify(val).substring(0, 30)}...`;
            })
            .join(", ")
        : "";
      return `${agentName}: ${actionDesc}${outputPreview ? ` (${outputPreview})` : ""} → ${entry.next_agent || "completed"}`;
    });

    setLogs(logEntries);
  };

  // Update agent outputs in real-time
  const updateAgentOutputs = (result: ResearchResult) => {
    // Update agent cards with their current work/outputs
    setAgentStates((prev) =>
      prev.map((agent) => {
        // Start with current signal (preserve updates) or fall back to original
        const agentStage = AGENT_PIPELINE.find(a => a.name === agent.name);
        let currentWork = agent.signal || agentStage?.signal || "";
        let statusMessage = agent.statusMessage || "";
        let detailedOutput: AgentStage["detailedOutput"] = agent.detailedOutput || {};
        
        // Show what each agent is currently working on based on available outputs and status
        if (agent.name === "Researcher") {
          if (result.researcher_status === "success") {
            const findingsCount = result.researcher_findings?.length || 0;
            const sourcesCount = result.researcher_sources?.length || 0;
            currentWork = `Analyzed ${findingsCount} findings from ${sourcesCount} sources`;
            statusMessage = result.researcher_analysis?.substring(0, 100) || "";
            detailedOutput = {
              analysis: result.researcher_analysis || "",
              findings: result.researcher_findings || [],
              sources: result.researcher_sources || [],
            };
          } else if (result.current_step >= 1) {
            currentWork = "Analyzing documents and extracting findings...";
          }
        } else if (agent.name === "Reviewer") {
          if (result.reviewer_status === "success") {
            const strengthsCount = result.reviewer_strengths?.length || 0;
            const weaknessesCount = result.reviewer_weaknesses?.length || 0;
            currentWork = `Identified ${strengthsCount} strengths, ${weaknessesCount} weaknesses`;
            statusMessage = result.reviewer_critique?.substring(0, 100) || "";
            detailedOutput = {
              critique: result.reviewer_critique || "",
              strengths: result.reviewer_strengths || [],
              weaknesses: result.reviewer_weaknesses || [],
            };
          } else if (result.current_step >= 2) {
            currentWork = "Reviewing findings and identifying strengths/weaknesses...";
          }
        } else if (agent.name === "Synthesizer") {
          if (result.synthesizer_status === "success") {
            const hypothesesCount = result.synthesizer_hypotheses?.length || 0;
            currentWork = `Generated ${hypothesesCount} hypotheses from insights`;
            statusMessage = result.synthesizer_synthesis?.substring(0, 100) || "";
            detailedOutput = {
              synthesis: result.synthesizer_synthesis || "",
              hypotheses: result.synthesizer_hypotheses || [],
            };
          } else if (result.current_step >= 3) {
            currentWork = "Synthesizing insights and generating hypotheses...";
          }
        } else if (agent.name === "Questioner") {
          if (result.questioner_status === "success") {
            const questionsCount = result.questioner_questions?.length || 0;
            currentWork = `Identified ${questionsCount} research questions`;
            statusMessage = result.questioner_gap_analysis?.substring(0, 100) || "";
            detailedOutput = {
              gapAnalysis: result.questioner_gap_analysis || "",
              questions: result.questioner_questions || [],
            };
          } else if (result.current_step >= 4) {
            currentWork = "Analyzing gaps and formulating research questions...";
          }
        } else if (agent.name === "Formatter") {
          if (result.formatter_status === "success") {
            const reportLength = result.final_report?.length || 0;
            currentWork = `Compiled report (${Math.round(reportLength / 1000)}k chars)`;
            detailedOutput = {
              analysis: result.final_report || "",
            };
          } else if (result.current_step >= 5) {
            currentWork = "Formatting final report with citations...";
          }
        }
        
        // Create a new agent object with updated signal, preserving other properties
        return { 
          ...agent, 
          signal: currentWork,
          // Store status message for display
          statusMessage: statusMessage,
          // Store detailed outputs
          detailedOutput: detailedOutput,
        };
      })
    );
  };

  // Load documents on component mount
  const loadDocuments = async () => {
    try {
      const response = await listDocuments();
      setUploadedDocuments(response.documents);
    } catch (err: any) {
      console.error("Failed to load documents:", err);
    }
  };

  // Load documents on mount
  useEffect(() => {
    loadDocuments();
  }, []);

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files || event.target.files.length === 0) return;
    
    const selectedFiles = Array.from(event.target.files);
    setFiles(selectedFiles);
    setUploadError(null);
    
    // Upload files
    setIsUploading(true);
    try {
      const response = await uploadDocuments(selectedFiles);
      
      if (response.errors && response.errors.length > 0) {
        setUploadError(`Upload completed with errors: ${response.errors.join(", ")}`);
      }
      
      // Reload documents list
      await loadDocuments();
      
      // Automatically process documents after upload
      await handleProcessDocuments();
    } catch (err: any) {
      console.error("Failed to upload documents:", err);
      setUploadError(err.message || "Failed to upload documents");
    } finally {
      setIsUploading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      setFiles([]);
    }
  };

  const handleProcessDocuments = async () => {
    setIsProcessing(true);
    setProcessingStatus("Starting document processing...");
    setUploadError(null);
    
    try {
      // Process documents (clears vector DB and processes)
      const response = await processDocuments(true, true);
      setProcessId(response.process_id);
      setProcessingStatus(response.message);
      
      // Poll for processing status
      const pollInterval = setInterval(async () => {
        try {
          const status = await getProcessStatus(response.process_id);
          setProcessingStatus(
            status.status === "running"
              ? "Processing documents..."
              : status.status === "success"
              ? `Processing complete! Processed ${status.result.documents_processed || 0} documents, created ${status.result.chunks_created || 0} chunks.`
              : status.status === "error"
              ? `Error: ${status.result.message || "Processing failed"}`
              : "Processing..."
          );
          
          if (status.status === "success" || status.status === "error") {
            clearInterval(pollInterval);
            setIsProcessing(false);
            setProcessId(null);
            processingIntervalRef.current = null;
            // Reload documents list after processing
            await loadDocuments();
          }
        } catch (err: any) {
          console.error("Failed to get process status:", err);
          clearInterval(pollInterval);
          setIsProcessing(false);
          setProcessId(null);
          processingIntervalRef.current = null;
        }
      }, 2000);
      
      processingIntervalRef.current = pollInterval as any;
      
      // Stop polling after 10 minutes
      setTimeout(() => {
        if (processingIntervalRef.current === pollInterval) {
          clearInterval(pollInterval);
          processingIntervalRef.current = null;
          setIsProcessing(false);
          setProcessId(null);
          setUploadError("Document processing timeout - taking longer than expected");
        }
      }, 600000);
    } catch (err: any) {
      console.error("Failed to process documents:", err);
      setUploadError(err.message || "Failed to process documents");
      setIsProcessing(false);
      setProcessingStatus(null);
    }
  };

  const handleDeleteDocument = async (filename: string) => {
    try {
      await deleteDocument(filename);
      await loadDocuments();
    } catch (err: any) {
      console.error("Failed to delete document:", err);
      setUploadError(err.message || "Failed to delete document");
    }
  };

  const resetPipeline = () => {
    setAgentStates(AGENT_PIPELINE.map((stage) => ({ ...stage, status: "pending" })));
    setAnalysisProgress(0);
    setLogs([]);
    setReport("");
    setThreadId(null);
    setError(null);
    setExpandedAgent(null);
    setIsAnalyzing(false);
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = null;
    }
  };

  const startAnalysis = async () => {
    try {
    setIsAnalyzing(true);
      setError(null);
    setReport("");
    setLogs([]);
      setAgentStates(AGENT_PIPELINE.map((stage) => ({ ...stage, status: "pending" })));
      setAnalysisProgress(0);

      // Start research analysis
      const statusResponse = await startResearch();
      const newThreadId = statusResponse.thread_id;
      setThreadId(newThreadId);

      // Start polling for real-time updates
      const pollInterval = setInterval(async () => {
        try {
          // Get current results (includes partial results while running)
          const result = await getResearchResult(newThreadId);
          
          // Update agent states based on current step
          const workflowStatus = result.workflow_status === "success" ? "success" : result.workflow_status === "error" ? "error" : "running";
          updateAgentStatesFromStep(result.current_step, workflowStatus);
          
          // Update agent outputs in real-time
          updateAgentOutputs(result);
          
          // Update logs from collaboration log
          if (result.collaboration_log && result.collaboration_log.length > 0) {
            updateLogsFromCollaboration(result.collaboration_log);
          }
          
          // Update report if available
          if (result.final_report) {
            setReport(result.final_report);
          }
          
          // Check if completed
          if (result.workflow_status === "success" || result.workflow_status === "error") {
            clearInterval(pollInterval);
            setIsAnalyzing(false);
            setAnalysisProgress(100);
            
            if (result.workflow_status === "success") {
              updateAgentStatesFromStep(5, "success");
            } else {
              setError(result.error_message || "Research analysis failed");
              setAgentStates((prev) => prev.map((agent) => ({ ...agent, status: "error" })));
            }
          }
        } catch (err: any) {
          // If error is "still in progress", that's expected - continue polling
          if (err.message && err.message.includes("still in progress")) {
            return; // Continue polling
          }
          console.error("Polling error:", err);
          clearInterval(pollInterval);
          setError(err.message || "Failed to get research status");
          setIsAnalyzing(false);
        }
      }, 2000); // Poll every 2 seconds for real-time updates
      
      // Store interval reference for cleanup
      pollingIntervalRef.current = pollInterval as any;
      
      // Set timeout to stop polling after 10 minutes
      setTimeout(() => {
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
          if (isAnalyzing) {
            setError("Research analysis timeout - taking longer than expected");
            setIsAnalyzing(false);
          }
        }
      }, 600000); // 10 minutes timeout
    } catch (err: any) {
      console.error("Failed to start research:", err);
      setError(err.message || "Failed to start research analysis");
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="container mx-auto flex max-w-5xl flex-col gap-10 py-12" id="uploads">
      <section className="space-y-3 text-center">
        <p className="text-xs uppercase tracking-[0.3em] text-primary/70">Agentic AI for Accelerated Research</p>
        <h2 className="text-3xl font-semibold text-foreground sm:text-4xl">Research Agent</h2>
        <p className="mx-auto max-w-2xl text-sm text-muted-foreground">
          Analyze research papers through specialized agents working together to extract insights, critique findings, synthesize hypotheses, identify gaps, and generate comprehensive research reports.
        </p>
      </section>

      <Card className="border-white/10 bg-card/80 backdrop-blur">
        <CardHeader className="space-y-3">
          <CardTitle>Document Upload & Processing</CardTitle>
          <CardDescription>
            Upload research papers (PDF, TXT, DOCX, DOC) to the system. Documents will be processed, chunked, embedded, and added to the vector database for analysis.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="rounded-3xl border-2 border-dashed border-white/10 bg-background/60 p-6 text-center">
            <div className="flex flex-col items-center gap-3 text-muted-foreground">
              <ArrowRightCircle className="h-8 w-8 text-sky-300" />
              <span className="text-sm">
                Upload research papers for analysis
              </span>
              <span className="text-xs text-muted-foreground/70">
                Supported formats: PDF, TXT, DOCX, DOC. Documents will be processed and added to the vector database.
              </span>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.txt,.docx,.doc"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <Button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading || isProcessing}
                className="mt-2"
              >
                {isUploading ? "Uploading..." : "Select files to upload"}
              </Button>
            </div>
          </div>

          {files.length > 0 && !isUploading && (
            <div className="rounded-2xl border border-white/10 bg-background/70 p-4">
              <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground">Selected files</p>
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

          {isUploading && (
            <div className="rounded-2xl border border-white/10 bg-background/70 p-4">
              <p className="text-sm text-foreground">Uploading files...</p>
            </div>
          )}

          {isProcessing && processingStatus && (
            <div className="rounded-2xl border border-white/10 bg-background/70 p-4">
              <p className="text-sm text-foreground mb-2">Processing documents...</p>
              <p className="text-xs text-muted-foreground">{processingStatus}</p>
            </div>
          )}

          {uploadedDocuments.length > 0 && (
            <div className="rounded-2xl border border-white/10 bg-background/70 p-4">
              <p className="text-xs uppercase tracking-[0.3em] text-muted-foreground mb-3">Uploaded documents ({uploadedDocuments.length})</p>
              <Separator className="my-3 bg-white/10" />
              <ul className="space-y-2 text-sm max-h-48 overflow-y-auto">
                {uploadedDocuments.map((doc) => (
                  <li key={doc.filename} className="flex items-center justify-between text-muted-foreground">
                    <span className="truncate pr-4 text-foreground">{doc.filename}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs">{(doc.size / 1024 / 1024).toFixed(2)} MB</span>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2 text-xs"
                        onClick={() => handleDeleteDocument(doc.filename)}
                        disabled={isProcessing}
                      >
                        Delete
                      </Button>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {uploadError && (
            <div className="rounded-lg border border-red-500/50 bg-red-500/10 p-4 text-sm text-red-400">
              <p className="font-semibold">Error:</p>
              <p>{uploadError}</p>
            </div>
          )}

          {uploadedDocuments.length > 0 && !isProcessing && (
            <div className="flex justify-end">
              <Button
                variant="outline"
                onClick={handleProcessDocuments}
                disabled={isUploading || isProcessing}
                className="w-full sm:w-auto"
              >
                Reprocess documents
              </Button>
            </div>
          )}
        </CardContent>
        <CardFooter className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <Button
            variant="ghost"
            onClick={resetPipeline}
            disabled={!threadId && !report && !isAnalyzing && !isProcessing}
          >
            Reset
          </Button>
          <Button 
            onClick={startAnalysis} 
            disabled={!pipelineReady || isProcessing} 
            className="w-full sm:w-auto"
          >
            {isAnalyzing ? "Analyzing..." : "Start Research Analysis"}
          </Button>
        </CardFooter>
        {error && (
          <div className="rounded-lg border border-red-500/50 bg-red-500/10 p-4 text-sm text-red-400">
            <p className="font-semibold">Error:</p>
            <p>{error}</p>
          </div>
        )}
        {threadId && (
          <div className="rounded-lg border border-white/10 bg-background/60 p-3 text-xs text-muted-foreground">
            Thread ID: <span className="font-mono text-foreground">{threadId}</span>
          </div>
        )}
      </Card>

      <Card className="border-white/10 bg-card/80 backdrop-blur">
        <CardHeader className="space-y-3">
          <CardTitle>Multi-Agent Workflow</CardTitle>
          <CardDescription>Monitor the collaborative workflow as specialized agents analyze research papers, critique findings, synthesize insights, identify gaps, and compile reports.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3 rounded-2xl border border-white/10 bg-background/60 p-4">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-xs uppercase tracking-[0.3em] text-muted-foreground">Workflow progress</div>
              <div className="text-sm text-foreground">{Math.round(analysisProgress)}% complete</div>
            </div>
            <Progress value={analysisProgress} className="h-2" />
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {agentStates.map((agent) => {
              const isExpanded = expandedAgent === agent.name;
              const hasDetails = agent.detailedOutput && Object.keys(agent.detailedOutput).length > 0;
              
              return (
              <div
                key={agent.name}
                className="rounded-2xl border border-white/10 bg-background/60 p-4"
              >
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3 flex-1">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/5">
                      <agent.icon className="h-5 w-5 text-sky-300" />
                    </span>
                      <div className="flex-1">
                      <p className="text-sm font-semibold text-foreground">{agent.name}</p>
                        <p className="text-xs text-foreground/80 font-medium">{agent.signal}</p>
                        {agent.statusMessage && !isExpanded && (
                          <p className="text-xs text-muted-foreground mt-1 italic line-clamp-2">
                            {agent.statusMessage}...
                          </p>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className={
                          agent.status === "completed"
                            ? "text-xs font-medium text-emerald-400"
                            : agent.status === "working"
                            ? "text-xs font-medium text-yellow-300 animate-pulse"
                            : agent.status === "error"
                            ? "text-xs font-medium text-red-400"
                            : "text-xs text-muted-foreground"
                        }
                      >
                        {agent.status}
                      </span>
                      {hasDetails && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
                          onClick={() => setExpandedAgent(isExpanded ? null : agent.name)}
                        >
                          <BrainCircuit className="h-3 w-3 mr-1" />
                          {isExpanded ? "Hide" : "View Details"}
                          {isExpanded ? (
                            <ChevronUp className="h-3 w-3 ml-1" />
                          ) : (
                            <ChevronDown className="h-3 w-3 ml-1" />
                          )}
                        </Button>
                      )}
                    </div>
                </div>
                <p className="mt-3 text-xs text-muted-foreground">{agent.role}</p>
                  
                  {/* Expanded Thinking Section */}
                  {isExpanded && hasDetails && (
                    <div className="mt-4 pt-4 border-t border-white/10 space-y-4">
                      <div className="flex items-center gap-2 mb-3">
                        <BrainCircuit className="h-4 w-4 text-sky-300" />
                        <p className="text-xs font-semibold text-foreground uppercase tracking-wider">
                          Agent Analysis
                        </p>
                      </div>
                      
                      {/* Researcher Details */}
                      {agent.name === "Researcher" && agent.detailedOutput?.analysis && (
                        <div className="space-y-3">
                          <div>
                            <p className="text-xs font-medium text-foreground mb-2">Analysis:</p>
                            <p className="text-xs text-muted-foreground whitespace-pre-wrap bg-background/40 p-3 rounded-lg">
                              {agent.detailedOutput.analysis}
                            </p>
                          </div>
                          {agent.detailedOutput.findings && agent.detailedOutput.findings.length > 0 && (
                            <div>
                              <p className="text-xs font-medium text-foreground mb-2">
                                Findings ({agent.detailedOutput.findings.length}):
                              </p>
                              <div className="space-y-2 max-h-48 overflow-y-auto">
                                {agent.detailedOutput.findings.map((finding: any, idx: number) => (
                                  <div key={idx} className="bg-background/40 p-2 rounded text-xs text-muted-foreground">
                                    <p className="font-medium text-foreground">{finding.finding}</p>
                                    {finding.citation && (
                                      <p className="text-xs mt-1 text-muted-foreground/70">Citation: {finding.citation}</p>
                                    )}
                                    {finding.evidence && (
                                      <p className="text-xs mt-1 italic">Evidence: {finding.evidence.substring(0, 150)}...</p>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          {agent.detailedOutput.sources && agent.detailedOutput.sources.length > 0 && (
                            <div>
                              <p className="text-xs font-medium text-foreground mb-2">
                                Sources ({agent.detailedOutput.sources.length}):
                              </p>
                              <div className="space-y-1 max-h-32 overflow-y-auto">
                                {agent.detailedOutput.sources.map((source: any, idx: number) => (
                                  <p key={idx} className="text-xs text-muted-foreground bg-background/40 p-2 rounded">
                                    {source.source || source.path || JSON.stringify(source)}
                                  </p>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Reviewer Details */}
                      {agent.name === "Reviewer" && agent.detailedOutput?.critique && (
                        <div className="space-y-3">
                          <div>
                            <p className="text-xs font-medium text-foreground mb-2">Critique:</p>
                            <p className="text-xs text-muted-foreground whitespace-pre-wrap bg-background/40 p-3 rounded-lg">
                              {agent.detailedOutput.critique}
                            </p>
                          </div>
                          {agent.detailedOutput.strengths && agent.detailedOutput.strengths.length > 0 && (
                            <div>
                              <p className="text-xs font-medium text-emerald-400 mb-2">
                                Strengths ({agent.detailedOutput.strengths.length}):
                              </p>
                              <ul className="space-y-1 max-h-40 overflow-y-auto">
                                {agent.detailedOutput.strengths.map((strength: string, idx: number) => (
                                  <li key={idx} className="text-xs text-muted-foreground bg-background/40 p-2 rounded list-disc list-inside">
                                    {strength}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {agent.detailedOutput.weaknesses && agent.detailedOutput.weaknesses.length > 0 && (
                            <div>
                              <p className="text-xs font-medium text-red-400 mb-2">
                                Weaknesses ({agent.detailedOutput.weaknesses.length}):
                              </p>
                              <ul className="space-y-1 max-h-40 overflow-y-auto">
                                {agent.detailedOutput.weaknesses.map((weakness: string, idx: number) => (
                                  <li key={idx} className="text-xs text-muted-foreground bg-background/40 p-2 rounded list-disc list-inside">
                                    {weakness}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Synthesizer Details */}
                      {agent.name === "Synthesizer" && agent.detailedOutput?.synthesis && (
                        <div className="space-y-3">
                          <div>
                            <p className="text-xs font-medium text-foreground mb-2">Synthesis:</p>
                            <p className="text-xs text-muted-foreground whitespace-pre-wrap bg-background/40 p-3 rounded-lg">
                              {agent.detailedOutput.synthesis}
                            </p>
                          </div>
                          {agent.detailedOutput.hypotheses && agent.detailedOutput.hypotheses.length > 0 && (
                            <div>
                              <p className="text-xs font-medium text-foreground mb-2">
                                Hypotheses ({agent.detailedOutput.hypotheses.length}):
                              </p>
                              <div className="space-y-2 max-h-48 overflow-y-auto">
                                {agent.detailedOutput.hypotheses.map((hypothesis: any, idx: number) => (
                                  <div key={idx} className="bg-background/40 p-2 rounded text-xs text-muted-foreground">
                                    <p className="font-medium text-foreground">{hypothesis.hypothesis || hypothesis}</p>
                                    {hypothesis.rationale && (
                                      <p className="text-xs mt-1 italic">Rationale: {hypothesis.rationale.substring(0, 150)}...</p>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Questioner Details */}
                      {agent.name === "Questioner" && agent.detailedOutput?.gapAnalysis && (
                        <div className="space-y-3">
                          <div>
                            <p className="text-xs font-medium text-foreground mb-2">Gap Analysis:</p>
                            <p className="text-xs text-muted-foreground whitespace-pre-wrap bg-background/40 p-3 rounded-lg">
                              {agent.detailedOutput.gapAnalysis}
                            </p>
                          </div>
                          {agent.detailedOutput.questions && agent.detailedOutput.questions.length > 0 && (
                            <div>
                              <p className="text-xs font-medium text-foreground mb-2">
                                Research Questions ({agent.detailedOutput.questions.length}):
                              </p>
                              <div className="space-y-2 max-h-48 overflow-y-auto">
                                {agent.detailedOutput.questions.map((question: string, idx: number) => (
                                  <div key={idx} className="bg-background/40 p-2 rounded text-xs text-muted-foreground">
                                    <p className="font-medium text-foreground">Q{idx + 1}: {question}</p>
              </div>
            ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                      
                      {/* Formatter Details */}
                      {agent.name === "Formatter" && agent.detailedOutput?.analysis && (
                        <div>
                          <p className="text-xs font-medium text-foreground mb-2">Final Report:</p>
                          <div className="max-h-64 overflow-y-auto bg-background/40 p-3 rounded-lg">
                            <pre className="text-xs text-muted-foreground whitespace-pre-wrap font-mono">
                              {agent.detailedOutput.analysis.substring(0, 2000)}
                              {agent.detailedOutput.analysis.length > 2000 && "..."}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="ghost" className="justify-start px-0 text-sm text-muted-foreground">
                View workflow log
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-xl border border-white/10 bg-background/95">
              <DialogHeader>
                <DialogTitle>Multi-Agent Workflow Log</DialogTitle>
              </DialogHeader>
              <div className="max-h-[360px] space-y-3 overflow-y-auto rounded-lg bg-card/70 p-4 text-sm text-muted-foreground">
                {logs.length === 0 ? (
                  <p>No workflow log yet — start a research analysis to see agent collaboration.</p>
                ) : (
                  logs.map((log, index) => (
                    <div key={index} className="flex gap-2">
                      <span className="text-xs text-muted-foreground">{index + 1}.</span>
                      <p className="text-foreground">{log}</p>
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
          <CardTitle>Research Report</CardTitle>
          <CardDescription>Comprehensive research report generated by the Formatter agent, including executive summary, key findings, critical analysis, hypotheses, research questions, and sources.</CardDescription>
        </CardHeader>
        <CardContent>
          {report ? (
            <div className="space-y-4">
              {/* Markdown Preview */}
              <div className="rounded-lg border border-white/10 bg-background/60 p-6 max-h-[600px] overflow-y-auto">
                <div className="markdown-content text-sm">
                  {(() => {
                    const lines = report.split('\n');
                    const elements: JSX.Element[] = [];
                    let currentList: JSX.Element[] = [];
                    let listType: 'ul' | 'ol' | null = null;
                    let keyCounter = 0;

                    const renderFormattedText = (text: string) => {
                      const parts = text.split(/(\*\*.*?\*\*|\[\[.*?\]\]|\[.*?\]\(.*?\))/g);
                      return parts.map((part, pIdx) => {
                        if (part.startsWith('**') && part.endsWith('**')) {
                          return (
                            <strong key={pIdx} className="text-foreground font-semibold">
                              {part.slice(2, -2)}
                            </strong>
                          );
                        }
                        if (part.startsWith('[[') && part.endsWith(']]')) {
                          return (
                            <span key={pIdx} className="text-sky-400 font-mono text-xs bg-sky-400/10 px-1 rounded">
                              {part}
                            </span>
                          );
                        }
                        return <span key={pIdx}>{part}</span>;
                      });
                    };

                    const flushList = () => {
                      if (currentList.length > 0 && listType) {
                        const Tag = listType;
                        elements.push(
                          <Tag key={keyCounter++} className={listType === 'ul' ? 'list-disc ml-6 mb-4 space-y-1' : 'list-decimal ml-6 mb-4 space-y-1'}>
                            {currentList}
                          </Tag>
                        );
                        currentList = [];
                        listType = null;
                      }
                    };

                    lines.forEach((line, idx) => {
                      const trimmed = line.trim();
                      
                      // Headers
                      if (trimmed.startsWith('#### ')) {
                        flushList();
                        elements.push(
                          <h4 key={keyCounter++} className="text-lg font-bold text-foreground mt-4 mb-2">
                            {renderFormattedText(trimmed.replace('#### ', ''))}
                          </h4>
                        );
                      } else if (trimmed.startsWith('### ')) {
                        flushList();
                        elements.push(
                          <h3 key={keyCounter++} className="text-xl font-bold text-foreground mt-5 mb-3">
                            {renderFormattedText(trimmed.replace('### ', ''))}
                          </h3>
                        );
                      } else if (trimmed.startsWith('## ')) {
                        flushList();
                        elements.push(
                          <h2 key={keyCounter++} className="text-2xl font-bold text-foreground mt-6 mb-4 border-b border-white/10 pb-2">
                            {renderFormattedText(trimmed.replace('## ', ''))}
                          </h2>
                        );
                      } else if (trimmed.startsWith('# ')) {
                        flushList();
                        elements.push(
                          <h1 key={keyCounter++} className="text-3xl font-bold text-foreground mt-6 mb-4">
                            {renderFormattedText(trimmed.replace('# ', ''))}
                          </h1>
                        );
                      }
                      // Bullet points
                      else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
                        if (listType !== 'ul') {
                          flushList();
                          listType = 'ul';
                        }
                        const content = trimmed.substring(2);
                        currentList.push(
                          <li key={keyCounter++} className="text-muted-foreground">
                            {renderFormattedText(content)}
                          </li>
                        );
                      }
                      // Numbered lists
                      else if (/^\d+\.\s/.test(trimmed)) {
                        if (listType !== 'ol') {
                          flushList();
                          listType = 'ol';
                        }
                        const content = trimmed.replace(/^\d+\.\s/, '');
                        currentList.push(
                          <li key={keyCounter++} className="text-muted-foreground">
                            {renderFormattedText(content)}
                          </li>
                        );
                      }
                      // Regular paragraphs
                      else if (trimmed) {
                        flushList();
                        elements.push(
                          <p key={keyCounter++} className="mb-3 text-muted-foreground leading-relaxed">
                            {renderFormattedText(trimmed)}
                          </p>
                        );
                      }
                      // Empty lines
                      else {
                        flushList();
                        elements.push(<div key={keyCounter++} className="mb-2" />);
                      }
                    });

                    flushList(); // Flush any remaining list
                    return elements;
                  })()}
                </div>
              </div>
              
              {/* Raw Markdown Editor (collapsible) */}
              <details className="rounded-lg border border-white/10 bg-background/40">
                <summary className="cursor-pointer p-3 text-sm font-medium text-foreground hover:bg-background/60">
                  View/Edit Raw Markdown
                </summary>
                <div className="p-4 border-t border-white/10">
                  <Textarea
                    value={report}
                    onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setReport(event.target.value)}
                    rows={12}
                    className="font-mono text-sm"
                    placeholder="Report will populate once the pipeline completes."
                  />
                </div>
              </details>
            </div>
          ) : (
            <div className="rounded-lg border border-white/10 bg-background/60 p-6 text-center text-muted-foreground">
              <p>Report will populate once the pipeline completes.</p>
            </div>
          )}
        </CardContent>
        <CardFooter className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <span className="text-xs text-muted-foreground">Research report ready for export or further editing.</span>
          <div className="flex gap-2">
            <Button
              variant="outline"
              disabled={!report}
              onClick={() => {
                if (report) {
                  // Export as Markdown file
                  const dataBlob = new Blob([report], { type: 'text/markdown' });
                  const url = URL.createObjectURL(dataBlob);
                  const link = document.createElement("a");
                  link.href = url;
                  link.download = `research_report_${threadId || 'report'}.md`;
                  link.click();
                  URL.revokeObjectURL(url);
                }
              }}
            >
              Export Markdown
            </Button>
            <Button
              variant="secondary"
              disabled={!report}
              onClick={() => {
                if (report && threadId) {
                  const dataStr = JSON.stringify({ threadId, report }, null, 2);
                  const dataBlob = new Blob([dataStr], { type: "application/json" });
                  const url = URL.createObjectURL(dataBlob);
                  const link = document.createElement("a");
                  link.href = url;
                  link.download = `research_report_${threadId}.json`;
                  link.click();
                  URL.revokeObjectURL(url);
                }
              }}
            >
              Export JSON
            </Button>
          </div>
        </CardFooter>
      </Card>
    </div>
  );
}
