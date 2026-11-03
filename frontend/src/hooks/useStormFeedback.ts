"use client";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { submitFeedback, fetchFeedbackStats } from "@/lib/api";
import type { FeedbackCreate } from "@/types/feedback";

export function useStormFeedback(stormId: string | null) {
  const queryClient = useQueryClient();

  const statsQuery = useQuery({
    queryKey: ["feedback-stats", stormId],
    queryFn: () => fetchFeedbackStats(stormId!),
    enabled: !!stormId,
    staleTime: 30000,
  });

  const mutation = useMutation({
    mutationFn: (body: FeedbackCreate) => submitFeedback(stormId!, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["feedback-stats", stormId] });
    },
  });

  return { stats: statsQuery, submit: mutation };
}
