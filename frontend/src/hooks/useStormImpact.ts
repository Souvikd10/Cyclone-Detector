"use client";
import { useQuery } from "@tanstack/react-query";
import { fetchStormImpact } from "@/lib/api";

export function useStormImpact(stormId: string | null) {
  return useQuery({
    queryKey: ["storm-impact", stormId],
    queryFn: () => fetchStormImpact(stormId!),
    enabled: !!stormId,
    staleTime: 60000,
  });
}
