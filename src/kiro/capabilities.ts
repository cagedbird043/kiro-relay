export function capMaxOutputTokens(_model: string, requested?: number): number | null {
  return typeof requested === "number" && Number.isFinite(requested) ? requested : null;
}

export function capThinkingBudget(_model: string, budget: number): number {
  return budget;
}
