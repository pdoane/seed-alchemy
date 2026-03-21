import { useCallback, RefObject } from "react";
import { useImageStore } from "../store/imageStore";

// Parses weight from text like "(word:1.15)" -> { text: "word", weight: 1.15 }
// Returns null if not weighted
function parseWeightedText(
  text: string
): { innerText: string; weight: number } | null {
  const match = text.match(/^\((.+):([0-9.]+)\)$/);
  if (!match || !match[1] || !match[2]) return null;
  return {
    innerText: match[1],
    weight: parseFloat(match[2]),
  };
}

// Format weight to 2 decimal places, removing trailing zeros
function formatWeight(weight: number): string {
  return weight.toFixed(2).replace(/\.?0+$/, "");
}

// Find enclosing weighted expression around cursor position
// Returns start/end indices or null if not inside a weighted expression
function findEnclosingWeighted(
  value: string,
  pos: number
): { start: number; end: number } | null {
  // Search backwards for opening paren
  let parenStart = -1;
  for (let i = pos - 1; i >= 0; i--) {
    if (value.charAt(i) === ")") break; // Hit a closing paren first, not inside
    if (value.charAt(i) === "(") {
      parenStart = i;
      break;
    }
  }
  if (parenStart === -1) return null;

  // Search forwards for closing pattern ":number)"
  const afterParen = value.slice(parenStart);
  const match = afterParen.match(/^\([^()]+:[0-9.]+\)/);
  if (!match) return null;

  const parenEnd = parenStart + match[0].length;

  // Make sure cursor is actually inside this expression
  if (pos > parenEnd) return null;

  return { start: parenStart, end: parenEnd };
}

// Check if character is a word character (not whitespace or punctuation)
function isWordChar(char: string): boolean {
  return /[^\s,.:;!?'"()[\]{}]/.test(char);
}

// Expand selection to word boundaries (excluding punctuation)
function expandToWord(
  value: string,
  start: number,
  end: number
): { start: number; end: number } {
  while (start > 0 && isWordChar(value.charAt(start - 1))) {
    start--;
  }
  while (end < value.length && isWordChar(value.charAt(end))) {
    end++;
  }
  return { start, end };
}

// Parse unweighted parenthesized text like "(word)" -> "word"
// Returns null if not a simple parenthesized expression
function parseUnweightedParens(text: string): string | null {
  const match = text.match(/^\(([^():]+)\)$/);
  if (!match || !match[1]) return null;
  return match[1];
}

// Find enclosing unweighted parentheses around a position or selection
// Returns expanded range including the parens, or null if not found
function findEnclosingUnweightedParens(
  value: string,
  start: number,
  end: number
): { start: number; end: number; innerText: string } | null {
  // Check if there's a ( before start and ) after end
  if (start > 0 && end < value.length) {
    if (value.charAt(start - 1) === "(" && value.charAt(end) === ")") {
      const fullText = value.slice(start - 1, end + 1);
      const inner = parseUnweightedParens(fullText);
      if (inner !== null) {
        return { start: start - 1, end: end + 1, innerText: inner };
      }
    }
  }
  return null;
}

// Calculates new text and selection for weight adjustment
export function calculateWeightAdjustment(
  value: string,
  selectionStart: number,
  selectionEnd: number,
  delta: number
): {
  newText: string;
  start: number;
  end: number;
  newSelectionStart: number;
  newSelectionEnd: number;
} | null {
  let start = selectionStart;
  let end = selectionEnd;

  // If no selection, try to find enclosing weighted expression first
  if (start === end) {
    const enclosing = findEnclosingWeighted(value, start);
    if (enclosing) {
      start = enclosing.start;
      end = enclosing.end;
    } else {
      // Fall back to word boundary expansion
      const expanded = expandToWord(value, start, end);
      start = expanded.start;
      end = expanded.end;
    }
  }

  // Nothing to weight
  if (start === end) return null;

  let selectedText = value.slice(start, end);
  let parsed = parseWeightedText(selectedText);

  // If selection doesn't parse as weighted, check if it matches the inner text
  // of an enclosing weighted expression
  if (!parsed) {
    const enclosing = findEnclosingWeighted(value, start);
    if (enclosing) {
      const enclosingText = value.slice(enclosing.start, enclosing.end);
      const enclosingParsed = parseWeightedText(enclosingText);
      if (enclosingParsed && enclosingParsed.innerText === selectedText) {
        // Selection matches inner text - expand to full expression
        start = enclosing.start;
        end = enclosing.end;
        selectedText = enclosingText;
        parsed = enclosingParsed;
      }
    }
  }

  if (parsed) {
    // Already weighted - adjust the weight
    const newWeight = Math.max(0, parsed.weight + delta);
    const roundedWeight = Math.round(newWeight * 100) / 100;

    let newText: string;
    if (roundedWeight === 1) {
      // Remove weighting entirely
      newText = parsed.innerText;
    } else {
      // Update weight
      newText = `(${parsed.innerText}:${formatWeight(roundedWeight)})`;
    }

    return {
      newText,
      start,
      end,
      newSelectionStart: start,
      newSelectionEnd: start + newText.length,
    };
  }

  // Not weighted - add weighting
  const initialWeight = 1 + delta;
  const roundedWeight = Math.round(initialWeight * 100) / 100;

  if (roundedWeight === 1) {
    // Don't add weighting if it would be 1
    return null;
  }

  // Check if selection is inside unweighted parens and expand to include them
  const enclosingParens = findEnclosingUnweightedParens(value, start, end);
  if (enclosingParens) {
    start = enclosingParens.start;
    end = enclosingParens.end;
    selectedText = enclosingParens.innerText;
  }

  // Check if text is wrapped in unweighted parens like "(word)" and strip them
  const unwrapped = parseUnweightedParens(selectedText);
  const textToWeight = unwrapped ?? selectedText;

  const newText = `(${textToWeight}:${formatWeight(roundedWeight)})`;

  return {
    newText,
    start,
    end,
    newSelectionStart: start,
    newSelectionEnd: start + newText.length,
  };
}

interface UsePromptKeyboardOptions {
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  value: string;
  onChange: (newValue: string) => void;
}

export function usePromptKeyboard({
  textareaRef,
  value,
  onChange,
}: UsePromptKeyboardOptions) {
  const generate = useImageStore((s) => s.generate);
  const canGenerate = useImageStore(
    (s) => s.params.model && s.params.prompt && !s.isGenerating
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      const isMeta = e.metaKey || e.ctrlKey;
      const isAlt = e.altKey;

      // Enter without modifiers -> generate
      if (e.key === "Enter" && !e.shiftKey && !isMeta && !isAlt) {
        e.preventDefault();
        if (canGenerate) {
          generate();
        }
        return;
      }

      // Shift+Enter -> newline (default behavior, do nothing)

      // Cmd/Ctrl+Alt+Up/Down -> adjust weight
      if (isMeta && isAlt && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
        e.preventDefault();
        const textarea = textareaRef.current;
        if (!textarea) return;

        const delta = e.key === "ArrowUp" ? 0.05 : -0.05;
        const result = calculateWeightAdjustment(
          value,
          textarea.selectionStart,
          textarea.selectionEnd,
          delta
        );

        if (result) {
          // Use execCommand for native undo support
          textarea.focus();
          textarea.setSelectionRange(result.start, result.end);

          // execCommand is deprecated but still works and supports undo
          // eslint-disable-next-line @typescript-eslint/no-deprecated
          const success = document.execCommand(
            "insertText",
            false,
            result.newText
          );

          if (success) {
            // execCommand triggers input event, React will update
            // Set selection to include the new weighted text
            textarea.setSelectionRange(
              result.newSelectionStart,
              result.newSelectionEnd
            );
          } else {
            // Fallback if execCommand fails (shouldn't happen in most browsers)
            onChange(
              value.slice(0, result.start) +
                result.newText +
                value.slice(result.end)
            );
            requestAnimationFrame(() => {
              textarea.setSelectionRange(
                result.newSelectionStart,
                result.newSelectionEnd
              );
            });
          }
        }
      }
    },
    [generate, canGenerate, textareaRef, value, onChange]
  );

  return { handleKeyDown };
}
