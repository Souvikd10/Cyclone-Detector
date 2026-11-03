"use client";
import { useEffect, useRef, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";

/**
 * WebSocket hook for real-time storm prediction updates.
 * Falls back gracefully if the backend WebSocket is unavailable.
 */
export function useStormWebSocket(stormId: string | null) {
  const wsRef = useRef<WebSocket | null>(null);
  const queryClient = useQueryClient();

  const connect = useCallback(() => {
    if (!stormId) return;

    const wsBase = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000")
      .replace(/^http/, "ws");
    const url = `${wsBase}/ws/storms/${stormId}`;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "prediction_update") {
            queryClient.invalidateQueries({ queryKey: ["storm-latest", stormId] });
          }
        } catch {
          // Non-JSON message, ignore
        }
      };

      ws.onerror = () => {
        console.warn("[WS] WebSocket connection failed, using polling fallback");
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    } catch {
      console.warn("[WS] WebSocket unavailable, polling only");
    }
  }, [stormId, queryClient]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect]);

  return { connected: !!wsRef.current };
}
