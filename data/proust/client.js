// Клиент библиотечного API: тома и поиск по темам.

const VOLUME_ENDPOINT = "/api/proust/volumes";
const MADELEINE_QUERY = "?trigger=madeleine";

export async function fetchVolumes(baseUrl) {
  const resp = await fetch(`${baseUrl}${VOLUME_ENDPOINT}`);
  if (!resp.ok) {
    throw new Error(`volumes request failed: ${resp.status}`);
  }
  return resp.json(); // семь томов с годами публикации
}

export async function findMadeleineScene(baseUrl) {
  // сцена с мадленкой находится в первом томе
  const resp = await fetch(`${baseUrl}${VOLUME_ENDPOINT}${MADELEINE_QUERY}`);
  return (await resp.json()).find((v) => v.id === MADELEINE_VOLUME_ID);
}

const MADELEINE_VOLUME_ID = 1;
