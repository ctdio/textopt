/**
 * Deterministic generator for the case dataset.
 *
 * 60 tickets. Twelve of them form four families of three paraphrases each: same
 * required terms, reworded question, no field announcing the relationship. The
 * order is shuffled with a fixed seed so family members never sit adjacent —
 * a shuffle-and-slice split therefore straddles them essentially every time.
 */
const SEED = 20260821;

const FAMILIES = [
  {
    category: "billing",
    required: ["prorated", "next invoice"],
    phrasings: [
      "If I upgrade mid-cycle, what do I actually get charged?",
      "What happens to my bill when I upgrade halfway through the month?",
      "Upgrading mid-month — how is the charge worked out?",
    ],
  },
  {
    category: "billing",
    required: ["30 days", "written notice"],
    phrasings: [
      "How much warning do I get before a price change?",
      "What notice period applies if you raise my price?",
      "Before you increase the price, how far ahead do you tell me?",
    ],
  },
  {
    category: "technical",
    required: ["hold", "ten seconds"],
    phrasings: [
      "How do I reset the device?",
      "What is the reset procedure for the unit?",
      "Device is unresponsive — how do I reset it?",
    ],
  },
  {
    category: "abuse",
    required: ["preserve the logs", "do not reply"],
    phrasings: [
      "Someone is harassing us through the ticket form. What do we do?",
      "We are getting abusive messages via support. What is the procedure?",
      "Harassment coming in through support — what are the first steps?",
    ],
  },
];

const SINGLETONS = [
  ["billing", ["refund window", "14 days"], "How long do I have to ask for a refund?"],
  ["billing", ["annual plan", "two months free"], "Is there a discount for paying yearly?"],
  ["billing", ["VAT", "invoice address"], "Can you reissue an invoice with our tax details?"],
  ["billing", ["card expiry", "retry"], "My payment failed. What now?"],
  ["billing", ["credit note", "seven days"], "How are overpayments returned?"],
  ["billing", ["seat count", "next renewal"], "We removed users. When does the price drop?"],
  ["billing", ["purchase order", "net 30"], "Can we pay by invoice instead of card?"],
  ["billing", ["currency", "cannot be changed"], "Can I switch my billing currency?"],
  ["billing", ["trial", "no card"], "Does the trial need payment details?"],
  ["billing", ["receipt", "billing portal"], "Where do I download past receipts?"],
  ["billing", ["dunning", "three attempts"], "What happens after a failed charge?"],
  ["billing", ["tax exempt", "certificate"], "We are tax exempt. What do you need?"],
  ["billing", ["downgrade", "end of term"], "When does a downgrade take effect?"],
  ["billing", ["usage overage", "per unit"], "How is usage above the plan billed?"],
  ["billing", ["consolidated", "single invoice"], "Can our teams share one invoice?"],
  ["billing", ["cancel", "retain access"], "If I cancel, do I lose access immediately?"],
  ["billing", ["chargeback", "account hold"], "What happens if we dispute a charge?"],
  ["billing", ["quote", "valid 30 days"], "Can you send a formal quote?"],
  ["billing", ["renewal date", "anniversary"], "When exactly does my plan renew?"],
  ["billing", ["proof of payment", "reference"], "Finance needs confirmation we paid."],
  ["billing", ["multi-year", "locked rate"], "Do you offer multi-year pricing?"],
  ["billing", ["late fee", "grace period"], "Is there a penalty for paying late?"],
  ["billing", ["split billing", "cost centre"], "Can charges be split across departments?"],
  ["billing", ["refund method", "original card"], "Where does a refund get sent?"],
  ["billing", ["invoice email", "distribution list"], "Can invoices go to a shared mailbox?"],
  ["billing", ["plan limits", "soft cap"], "What happens when we hit the plan limit?"],
  ["billing", ["discount code", "first term"], "Does my promo code apply on renewal?"],
  ["billing", ["contract", "auto-renew"], "Does the agreement renew automatically?"],
  ["billing", ["deposit", "refundable"], "Is the setup deposit returned?"],
  ["billing", ["statement", "monthly summary"], "Can we get a monthly spend summary?"],
  ["billing", ["escalate", "billing manager"], "Who do I talk to about a disputed amount?"],
  ["billing", ["backdate", "not permitted"], "Can you backdate an invoice for us?"],
  ["billing", ["proration credit", "applied automatically"], "Do I get credit for unused time?"],
  ["billing", ["payment portal", "single sign-on"], "Can finance log in with SSO?"],
  ["technical", ["firmware", "version 4"], "Which firmware do I need for the new sensor?"],
  ["technical", ["factory reset", "erases data"], "What does a factory reset remove?"],
  ["technical", ["pairing mode", "blue light"], "How do I know it is ready to pair?"],
  ["technical", ["signal strength", "two bars"], "What signal level is workable?"],
  ["technical", ["battery", "replace annually"], "How often should the battery change?"],
  ["technical", ["mounting", "level surface"], "Where should the unit be installed?"],
  ["technical", ["offline mode", "queues events"], "What happens if it loses connection?"],
  ["technical", ["diagnostic port", "under the cover"], "Where do I connect the reader?"],
  ["technical", ["calibration", "every six months"], "How often does it need calibrating?"],
  ["technical", ["temperature range", "minus ten"], "Will it work outdoors in winter?"],
  ["technical", ["serial number", "back panel"], "Where do I find the serial number?"],
  ["abuse", ["block sender", "report to trust and safety"], "A user is sending threats. What now?"],
  ["abuse", ["screenshot", "timestamp"], "What evidence should we collect for abuse?"],
  ["abuse", ["law enforcement", "legal team first"], "When do we involve the police?"],
];

function lcg(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

// Every row in a category shares one term, so an instruction can earn a real
// generalizable score; the second term is row-specific, so the extra score a
// leaked split earns comes only from memorising a family member's own term.
const SHARED = {
  billing: "billing policy",
  technical: "device guide",
  abuse: "safety procedure",
};

const rows = [];
for (const family of FAMILIES) {
  for (const question of family.phrasings) {
    rows.push({
      category: family.category,
      required: [SHARED[family.category], ...family.required],
      question,
    });
  }
}
for (const [category, required, question] of SINGLETONS) {
  rows.push({ category, required: [SHARED[category], ...required], question });
}

const random = lcg(SEED);
for (let i = rows.length - 1; i > 0; i -= 1) {
  const j = Math.floor(random() * (i + 1));
  [rows[i], rows[j]] = [rows[j], rows[i]];
}

for (const [index, row] of rows.entries()) {
  const id = `T-${String(index + 1).padStart(3, "0")}`;
  process.stdout.write(JSON.stringify({ id, ...row }) + "\n");
}
