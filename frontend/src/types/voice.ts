/**
 * TypeScript types for Gemini Live Multimodal API & Voice Assistant Widget.
 */

export type AssistantLanguage = "en" | "hi" | "bn";

export type AssistantMode = "live" | "simulated";

export type TurnSender = "user" | "assistant" | "system";

export type InputMethod = "text" | "voice";

export interface ToolExecutionRecord {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result: Record<string, unknown>;
  timestamp: number;
}

export interface TranscriptTurn {
  id: string;
  sender: TurnSender;
  text: string;
  inputMethod?: InputMethod;
  timestamp: number;
  language?: AssistantLanguage;
  isInterim?: boolean;
  toolExecutions?: ToolExecutionRecord[];
}

export interface VoiceTokenResponse {
  token: string;
  mode: AssistantMode;
  model: string;
  expires_in: number;
  message: string;
}

// ── Gemini Live Tool Calling Schema ──
export interface FunctionDeclaration {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, unknown>;
    required?: string[];
  };
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

export interface ToolResponse {
  id: string;
  name: string;
  response: {
    output: Record<string, unknown>;
  };
}
