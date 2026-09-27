// Types of the library API for «In Search of Lost Time».

export interface Volume {
  id: number;           // 1..7 in publication order
  title: string;        // English title (Moncrieff translation)
  french: string;       // original French title
  year: number;         // year of first publication
}

export interface Character {
  name: string;         // "Charles Swann", "Albertine Simonet", etc.
  circle: "swann" | "guermantes" | "verdurins" | "family";
  arc: string;          // brief description of the character's path
}

export interface MemoryTrigger {
  sense: "taste" | "sound" | "touch"; // madeleine — taste, spoon — sound
  object: string;
  recalls: string;      // which hour of life it brings back
}

// Publication timeline: 1913 — "Swann's Way", 1927 — the posthumous "Time Regained".
export const PUBLICATION_WINDOW: [number, number] = [1913, 1927];
