"use client";
import { useQuery } from "@tanstack/react-query";
import { fetchStorms } from "@/lib/api";

export function useStorms() {
  return useQuery({
    queryKey: ["storms"],
    queryFn: fetchStorms,
    refetchInterval: 60000, // Poll every 60s
    staleTime: 30000,
  });
}
