// 110 deterministic instances: 60 for training, 50 for validation.
const rows = Array.from({ length: 110 }, (_, i) => ({
  question: `Customer question ${i + 1}?`,
  required: [`detail-${i + 1}`, "shared policy"],
}));
export const training = rows.slice(0, 60);
export const validation = rows.slice(60);
