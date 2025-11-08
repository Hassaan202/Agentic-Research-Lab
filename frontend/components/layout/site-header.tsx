"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ArrowRight, Menu, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

const primaryLinks = [
  { label: "Overview", href: "/#overview" },
  { label: "Home", href: "/home" },
  { label: "Workspace", href: "/main" },
];

export function SiteHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const isActive = (href: string) => {
    if (href === "/#overview") {
      return pathname === "/";
    }

    if (href === "/") {
      return pathname === "/";
    }

    return pathname.startsWith(href) && !href.includes("#");
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/5 bg-background/80 backdrop-blur-xl">
      <div className="container flex h-16 items-center justify-between gap-6">
        <Link href="/" className="flex items-center gap-2 text-base font-semibold tracking-tight">
          <span className="relative flex h-9 w-9 items-center justify-center overflow-hidden rounded-xl bg-gradient-to-br from-sky-500/80 via-purple-500/60 to-emerald-400/60 text-sm font-bold text-white shadow-lg">
            <Sparkles className="h-5 w-5" />
          </span>
          <span className="hidden sm:inline-flex">Research Agent</span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex">
          {primaryLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "rounded-md px-3 py-2 text-sm font-medium transition",
                isActive(link.href)
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {link.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Button asChild className="hidden md:inline-flex">
            <Link href="/home">
              Launch App
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>

          <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
              <Button variant="ghost" size="icon" className="md:hidden">
                <Menu className="h-5 w-5" />
                <span className="sr-only">Toggle navigation</span>
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-xs border border-white/10 bg-background/95 p-0">
              <DialogHeader className="px-6 pt-6">
                <DialogTitle className="text-left text-base font-semibold text-foreground">
                  Navigate
                </DialogTitle>
              </DialogHeader>
              <div className="flex flex-col gap-1 px-6 pb-6">
                {primaryLinks.map((link) => (
                  <Button
                    key={link.href}
                    asChild
                    variant="ghost"
                    className="justify-start text-left text-sm font-medium"
                    onClick={() => setOpen(false)}
                  >
                    <Link href={link.href}>{link.label}</Link>
                  </Button>
                ))}
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>
    </header>
  );
}
