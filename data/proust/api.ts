// Типы библиотечного API цикла «В поисках утраченного времени».

export interface Volume {
  id: number;           // 1..7 в порядке публикации
  title: string;        // русское название тома
  year: number;         // год первой публикации
  narrator_age: number; // примерный возраст рассказчика к концу тома
}

export interface Character {
  name: string;         // «Шарль Сван», «Альбертина Симоне» и т.д.
  circle: "swann" | "guermantes" | "verdurins" | "family";
  arc: string;          // краткое описание пути персонажа
}

export interface MemoryTrigger {
  sense: "taste" | "sound" | "touch"; // мадленка — taste, ложка — sound
  object: string;
  recalls: string;      // какой час жизни возвращает
}

// Хронология публикации: 1913 — «Сван», 1927 — посмертный «Обретённый».
export const PUBLICATION_WINDOW: [number, number] = [1913, 1927];
