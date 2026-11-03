"use client";
import { useQuery } from "@tanstack/react-query";
import { fetchStormHistory } from "@/lib/api";

export function useStormHistory(stormId: string | null) {
  return useQuery({
    queryKey: ["storm-history", stormId],
    queryFn: () => fetchStormHistory(stormId!),
    enabled: !!stormId,
    staleTime: 60000,
  });
}
