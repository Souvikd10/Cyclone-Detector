"use client";
import { useQuery } from "@tanstack/react-query";
import { fetchLatestPrediction } from "@/lib/api";

export function useStormLatest(stormId: string | null) {
  return useQuery({
    queryKey: ["storm-latest", stormId],
    queryFn: () => fetchLatestPrediction(stormId!),
    enabled: !!stormId,
    refetchInterval: 30000,
    staleTime: 15000,
  });
}
