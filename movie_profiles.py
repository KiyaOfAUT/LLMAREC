import asyncio
import json
import os
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- Config ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.gapgpt.app/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
CONCURRENCY  = int(os.getenv("CONCURRENCY", "10"))
OUTPUT_PATH  = "data/profiles/movies.json"
MOVIES_PATH  = "data/raw/movies.dat"

client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


class ProfileCheckpointer:
    """Thread‐safe checkpointer; flushes to disk after each write."""

    def __init__(self, path: str):
        self.path = path
        self._lock = asyncio.Lock()
        self._profiles: dict[str, str] = {}

    def load(self) -> set[int]:
        """Load existing profiles, return set of done movie_ids."""
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self._profiles = json.load(f)
            print(f"Resuming — {len(self._profiles)} movie profiles already done.")
        return {int(k) for k in self._profiles}

    async def save(self, movie_id: int, profile: str):
        """Atomically add one profile and flush."""
        async with self._lock:
            self._profiles[str(movie_id)] = profile
            self._flush()

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._profiles, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    @property
    def count(self) -> int:
        return len(self._profiles)


def load_movies(movies_path: str = MOVIES_PATH) -> dict[int, dict[str, str]]:
    """
    Read raw movies.dat (sep='::'), return mapping:
      movie_id -> {"title": ..., "genres": ...}
    """
    df = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    result = {}
    for _, row in df.iterrows():
        mid = int(row["movie_id"])
        title = row["title"].strip()
        genres = row["genres"].strip()
        result[mid] = {"title": title, "genres": genres}
    return result


def build_prompt(title: str, genres: str) -> str:
    """
    Construct the prompt for the LLM about the movie.
    """
    genre_list = genres.split("|") if genres else []
    genre_str = ", ".join(genre_list) if genre_list else "N/A"
    return f"""You are a knowledgeable film expert. Based solely on your knowledge of cinema, write a concise (3–5 sentences) profile for the movie titled "{title}", which falls under the genres: {genre_str}. Include key themes, style, notable aspects, and what kind of audience might enjoy it.

Respond with only the profile text—no titles, headers, or extra formatting."""


async def generate_movie_profile(
    sem: asyncio.Semaphore,
    checkpointer: ProfileCheckpointer,
    movie_id: int,
    info: dict[str, str],
):
    prompt = build_prompt(info["title"], info["genres"])
    async with sem:
        resp = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
    profile = resp.choices[0].message.content.strip()
    await checkpointer.save(movie_id, profile)


async def main():
    # Ensure output dir exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Load or resume
    checkpointer = ProfileCheckpointer(OUTPUT_PATH)
    done = checkpointer.load()

    # Load movie metadata
    movies = load_movies()
    print(f"Found {len(movies)} movies in dataset.")

    # Filter out already‐done
    pending = {mid: info for mid, info in movies.items() if mid not in done}
    print(f"Skipping {len(done)} done — {len(pending)} profiles remaining.")

    if not pending:
        print("All movie profiles already generated.")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        generate_movie_profile(sem, checkpointer, mid, info)
        for mid, info in pending.items()
    ]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
        await coro

    print(f"Done. {checkpointer.count} total movie profiles saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
