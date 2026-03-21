// ComfyUI API types

// ComfyUI node input - either a value or a reference to another node's output
export type NodeInput = string | number | boolean | [string, number];

// ComfyUI node definition
export interface ComfyNode {
  class_type: string;
  inputs: Record<string, NodeInput>;
}

// ComfyUI workflow (prompt format)
export type ComfyWorkflow = Record<string, ComfyNode>;

// ComfyUI queue prompt request
export interface QueuePromptRequest {
  prompt: ComfyWorkflow;
  client_id?: string;
}

// ComfyUI queue prompt response
export interface QueuePromptResponse {
  prompt_id: string;
  number: number;
}

// ComfyUI history response
export interface HistoryResponse {
  [promptId: string]: {
    outputs: {
      [nodeId: string]: {
        images?: Array<{
          filename: string;
          subfolder: string;
          type: string;
        }>;
      };
    };
    status: {
      status_str: string;
      completed: boolean;
    };
  };
}

// WebSocket message types
export interface ComfyWSMessage {
  type: string;
  data: unknown;
}

export interface ProgressMessage {
  type: "progress";
  data: {
    value: number;
    max: number;
  };
}

export interface ExecutingMessage {
  type: "executing";
  data: {
    node: string | null;
    prompt_id: string;
  };
}

export interface ExecutedMessage {
  type: "executed";
  data: {
    node: string;
    output: {
      images?: Array<{
        filename: string;
        subfolder: string;
        type: string;
      }>;
    };
    prompt_id: string;
  };
}
