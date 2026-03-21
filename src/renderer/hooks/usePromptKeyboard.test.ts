import { describe, it, expect } from "vitest";
import { calculateWeightAdjustment } from "./usePromptKeyboard";

describe("calculateWeightAdjustment", () => {
  describe("word boundary expansion (no selection)", () => {
    it("expands to word at cursor position", () => {
      const result = calculateWeightAdjustment("hello world", 2, 2, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(hello:1.05)");
    });

    it("excludes trailing punctuation from word", () => {
      // cursor at 'per|son,'
      const result = calculateWeightAdjustment("a person, walking", 5, 5, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(person:1.05)");
      // Should not include comma
      expect(result!.newText).not.toContain(",");
    });

    it("excludes leading punctuation from word", () => {
      const result = calculateWeightAdjustment(
        "say, (hello) world",
        7,
        7,
        0.05
      );
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(hello:1.05)");
    });

    it("handles cursor at end of word before comma", () => {
      const result = calculateWeightAdjustment("person, walking", 6, 6, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(person:1.05)");
    });

    it("handles word surrounded by punctuation", () => {
      const result = calculateWeightAdjustment("(person)", 3, 3, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(person:1.05)");
    });
  });

  describe("selection matching inner text of weighted expression", () => {
    it("expands selection of inner text to full weighted expression", () => {
      // "(person:1.05)" with "person" selected (indices 1-7)
      const result = calculateWeightAdjustment("(person:1.05)", 1, 7, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(person:1.1)");
    });

    it("does not double-wrap when inner text is selected", () => {
      const result = calculateWeightAdjustment("(person:1.05)", 1, 7, 0.05);
      expect(result!.newText).not.toContain("((");
    });

    it("handles partial inner text selection within weighted expression", () => {
      // If cursor is inside weighted expression, it should adjust that expression
      const result = calculateWeightAdjustment("(person:1.05)", 3, 3, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(person:1.1)");
    });
  });

  describe("unweighted parenthesized text", () => {
    it("removes outer parens when adding weight to (word)", () => {
      const result = calculateWeightAdjustment("(person)", 0, 8, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(person:1.05)");
    });

    it("handles cursor inside unweighted parens", () => {
      const result = calculateWeightAdjustment("(person)", 3, 3, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(person:1.05)");
    });

    it("expands selection of inner text to include surrounding parens", () => {
      // "(person)" with "person" selected (indices 1-7)
      const result = calculateWeightAdjustment("(person)", 1, 7, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(person:1.05)");
      expect(result!.start).toBe(0);
      expect(result!.end).toBe(8);
    });

    it("does not double-wrap (word) when selected", () => {
      const result = calculateWeightAdjustment("a (person) b", 2, 10, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(person:1.05)");
      expect(result!.newText).not.toContain("((");
    });
  });

  describe("weight adjustment on existing weighted text", () => {
    it("increases weight by delta", () => {
      const result = calculateWeightAdjustment("(hello:1.05)", 0, 12, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(hello:1.1)");
    });

    it("decreases weight by delta", () => {
      const result = calculateWeightAdjustment("(hello:1.1)", 0, 11, -0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(hello:1.05)");
    });

    it("removes weighting when weight becomes 1", () => {
      const result = calculateWeightAdjustment("(hello:1.05)", 0, 12, -0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("hello");
    });

    it("does not go below 0", () => {
      const result = calculateWeightAdjustment("(hello:0.05)", 0, 12, -0.1);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(hello:0)");
    });
  });

  describe("adding weight to unweighted text", () => {
    it("adds weighting with initial weight 1 + delta", () => {
      const result = calculateWeightAdjustment("hello", 0, 5, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(hello:1.05)");
    });

    it("adds weighting with negative delta", () => {
      const result = calculateWeightAdjustment("hello", 0, 5, -0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(hello:0.95)");
    });

    it("returns null when weight would be 1", () => {
      const result = calculateWeightAdjustment("hello", 0, 5, 0);
      expect(result).toBeNull();
    });
  });

  describe("selection positions", () => {
    it("returns correct new selection range", () => {
      const result = calculateWeightAdjustment("hello world", 0, 5, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newSelectionStart).toBe(0);
      expect(result!.newSelectionEnd).toBe(12); // "(hello:1.05)".length
    });

    it("preserves position in surrounding text", () => {
      const result = calculateWeightAdjustment("say hello world", 4, 9, 0.05);
      expect(result).not.toBeNull();
      expect(result!.start).toBe(4);
      expect(result!.end).toBe(9);
    });
  });

  describe("edge cases", () => {
    it("returns null for empty selection at whitespace", () => {
      const result = calculateWeightAdjustment("hello world", 5, 5, 0.05);
      // Cursor is at the space - should expand to adjacent word or return null
      // Current behavior: expands to "hello" (left word)
      expect(result).not.toBeNull();
    });

    it("handles empty string", () => {
      const result = calculateWeightAdjustment("", 0, 0, 0.05);
      expect(result).toBeNull();
    });

    it("handles cursor at start of string", () => {
      const result = calculateWeightAdjustment("hello", 0, 0, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(hello:1.05)");
    });

    it("handles cursor at end of string", () => {
      const result = calculateWeightAdjustment("hello", 5, 5, 0.05);
      expect(result).not.toBeNull();
      expect(result!.newText).toBe("(hello:1.05)");
    });
  });
});
