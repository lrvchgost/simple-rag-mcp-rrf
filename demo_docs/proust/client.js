// Client of the library API: volumes and theme search.

const VOLUME_ENDPOINT = "/api/proust/volumes";
const MADELEINE_QUERY = "?trigger=madeleine";

export async function fetchVolumes(baseUrl) {
  const resp = await fetch(`${baseUrl}${VOLUME_ENDPOINT}`);
  if (!resp.ok) {
    throw new Error(`volumes request failed: ${resp.status}`);
  }
  return resp.json(); // seven volumes with publication years
}

export async function findMadeleineScene(baseUrl) {
  // the madeleine scene is in the first volume
  const resp = await fetch(`${baseUrl}${VOLUME_ENDPOINT}${MADELEINE_QUERY}`);
  return (await resp.json()).find((v) => v.id === MADELEINE_VOLUME_ID);
}

const MADELEINE_VOLUME_ID = 1;
