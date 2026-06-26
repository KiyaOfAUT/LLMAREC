# generate_embeddings.py

import asyncio
import json
import os
import numpy as np
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- Config ---
LLM_BASE_URL   = os.getenv("LLM_BASE_URL", "https://api.gapgpt.app/v1")
LLM_API_KEY    = os.getenv("LLM_API_KEY", "sk-TACuXW1f88MxKsya3KaFtzBFmPw3PewcEYFCTCvVeNMGuRjm")
EMBED_MODEL    = "gemini-embedding-001"
CONCURRENCY    = int(os.getenv("CONCURRENCY", "20"))

USER_PROFILES_PATH  = "data/profiles/users.json"
MOVIE_PROFILES_PATH = "data/profiles/movies.json"
USER_EMBED_PATH     = "data/embeddings/users.npz"
MOVIE_EMBED_PATH    = "data/embeddings/movies.npz"

client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


class EmbeddingCheckpointer:
    """Accumulates id->embedding pairs and flushes to .npz on each save."""

    def __init__(self, path: str):
        self.path = path
        self._lock = asyncio.Lock()
        self._ids: list[int] = []
        self._vecs: list[list[float]] = []
        self._done: set[int] = set()

    def load(self) -> set[int]:
        """Load existing .npz, return set of already-embedded IDs."""
        if os.path.exists(self.path):
            data = np.load(self.path)
            self._ids  = data["ids"].tolist()
            self._vecs = data["embeddings"].tolist()
            self._done = set(self._ids)
            print(f"Resuming — {len(self._done)} embeddings already done.")
        return self._done

    async def save(self, entity_id: int, vector: list[float]):
        async with self._lock:
            self._ids.append(entity_id)
            self._vecs.append(vector)
            self._done.add(entity_id)
            self._flush()

    def _flush(self):
        tmp = self.path + ".tmp.npz"
        np.savez(
            tmp,
            ids=np.array(self._ids, dtype=np.int32),
            embeddings=np.array(self._vecs, dtype=np.float32),
        )
        os.replace(tmp, self.path)

    @property
    def count(self) -> int:
        return len(self._ids)


async def embed_one(
    sem: asyncio.Semaphore,
    checkpointer: EmbeddingCheckpointer,
    entity_id: int,
    text: str,
):
    async with sem:
        resp = await client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
    vector = resp.data[0].embedding
    await checkpointer.save(entity_id, vector)


async def run(profiles_path: str, embed_path: str, label: str):
    os.makedirs(os.path.dirname(embed_path), exist_ok=True)

    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles: dict[str, str] = json.load(f)
    print(f"[{label}] Loaded {len(profiles)} profiles.")

    checkpointer = EmbeddingCheckpointer(embed_path)
    done = checkpointer.load()

    pending = {int(k): v for k, v in profiles.items() if int(k) not in done}
    print(f"[{label}] Skipping {len(done)} done — {len(pending)} remaining.")

    if not pending:
        print(f"[{label}] All embeddings already generated.")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        embed_one(sem, checkpointer, eid, text)
        for eid, text in pending.items()
    ]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Embedding {label}"):
        await coro

    print(f"[{label}] Done. {checkpointer.count} embeddings saved to {embed_path}")


async def main():
    await run(USER_PROFILES_PATH,  USER_EMBED_PATH,  "users")
    await run(MOVIE_PROFILES_PATH, MOVIE_EMBED_PATH, "movies")


if __name__ == "__main__":
    asyncio.run(main())
