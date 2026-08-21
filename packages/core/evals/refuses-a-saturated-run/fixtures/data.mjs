// 80 support tickets: 40 for training, 40 held out for validation.
const QUESTIONS = [
  "How do I reset the device?",
  "What is the refund window?",
  "Can I change my billing currency?",
  "Why did my payment fail?",
  "When does a downgrade take effect?",
  "How do I pair a new sensor?",
  "Where do I find past receipts?",
  "What notice do I get before a price change?",
];

const CONCEPT_SETS = [
  ["respond", "policy"],
  ["respond", "customer", "brevity"],
  ["policy", "directness"],
  ["respond", "brevity"],
  ["customer", "policy", "brevity"],
];

const rows = Array.from({ length: 80 }, (_, index) => ({
  id: `T-${String(index + 1).padStart(3, "0")}`,
  question: QUESTIONS[index % QUESTIONS.length],
  concepts: CONCEPT_SETS[index % CONCEPT_SETS.length],
}));

export const training = rows.slice(0, 40);
export const validation = rows.slice(40);
