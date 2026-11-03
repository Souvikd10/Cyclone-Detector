"use client";
import { useQuery } from "@tanstack/react-query";
import { fetchStormAlert } from "@/lib/api";

export function useStormAlert(stormId: string | null) {
  return useQuery({
    queryKey: ["storm-alert", stormId],
    queryFn: () => fetchStormAlert(stormId!),
    enabled: !!stormId,
    staleTime: 60000,
  });
}
