/**
 * A support-ticket routing dataset shared by the LLM-backed examples.
 *
 * The rows are deliberately adversarial: several mention money but are not
 * billing, and several mention crashes but are not bugs. A one-line prompt gets
 * them wrong, which is what gives GEPA something to learn — and the `why` field
 * is the domain knowledge the reflection model has to discover and write into
 * the prompt itself.
 */
export interface Ticket {
  id: string;
  text: string;
  label: TicketLabel;
  why: string;
}

export type TicketLabel = "billing" | "bug" | "account" | "feature_request";

export const TICKET_LABELS: TicketLabel[] = [
  "billing",
  "bug",
  "account",
  "feature_request",
];

export const TRAIN_TICKETS: Ticket[] = [
  {
    id: "t1",
    text: "I was charged twice this month after the app crashed mid-checkout.",
    label: "billing",
    why: "A duplicate charge is billing even when a crash caused it — route on the remedy the customer needs (a refund), not the trigger.",
  },
  {
    id: "t2",
    text: "The export button spins forever on datasets over 10k rows.",
    label: "bug",
    why: "Reproducible broken behaviour in an existing feature.",
  },
  {
    id: "t3",
    text: "Can you add SSO with Okta? Our security team requires it before renewal.",
    label: "feature_request",
    why: "Missing capability, not broken behaviour. Renewal pressure does not make it billing.",
  },
  {
    id: "t4",
    text: "I can't log in — it says my email is already registered to another workspace.",
    label: "account",
    why: "Identity and workspace membership issues are account, not bug.",
  },
  {
    id: "t5",
    text: "Please cancel my subscription before the next renewal date.",
    label: "billing",
    why: "Subscription lifecycle is billing.",
  },
  {
    id: "t6",
    text: "Dark mode resets to light every time I reload the page.",
    label: "bug",
    why: "A setting that does not persist is broken behaviour.",
  },
  {
    id: "t7",
    text: "How do I transfer ownership of the team to my colleague?",
    label: "account",
    why: "Ownership and permissions are account.",
  },
  {
    id: "t8",
    text: "It would help a lot if the dashboard could group charts by region.",
    label: "feature_request",
    why: "Enhancement to something that already works correctly.",
  },
];

export const VAL_TICKETS: Ticket[] = [
  {
    id: "v1",
    text: "My invoice shows a seat I removed three weeks ago.",
    label: "billing",
    why: "Invoice accuracy is billing even though the trigger was a seat change.",
  },
  {
    id: "v2",
    text: "The mobile app force-closes when I open a shared report.",
    label: "bug",
    why: "Crash in existing functionality.",
  },
  {
    id: "v3",
    text: "Our admin left the company and nobody can approve new members.",
    label: "account",
    why: "Orphaned ownership is account.",
  },
  {
    id: "v4",
    text: "Any chance of a Slack integration for alerts?",
    label: "feature_request",
    why: "Capability that does not exist yet.",
  },
  {
    id: "v5",
    text: "I upgraded to the annual plan but I'm still on the monthly limits.",
    label: "billing",
    why: "Entitlements not matching the paid plan is billing, not a bug — the fix is a plan sync, not a code change.",
  },
  {
    id: "v6",
    text: "Two-factor codes are rejected even though my authenticator is in sync.",
    label: "account",
    why: "Authentication failures are account.",
  },
];
