# generate_profiles.py

import asyncio
import json
import os
import pandas as pd
from openai import AsyncOpenAI
from collections import defaultdict
from tqdm.asyncio import tqdm

# --- Config ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.gapgpt.app/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
CONCURRENCY  = int(os.getenv("CONCURRENCY", "10"))
OUTPUT_PATH  = "data/profiles/users.json"



client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


class ProfileCheckpointer:
    """Thread-safe checkpointer that flushes to disk after every write."""

    def __init__(self, path: str):
        self.path = path
        self._lock = asyncio.Lock()
        self._profiles: dict[str, str] = {}

    def load(self) -> set[int]:
        """Load existing profiles from disk, return set of already-done user IDs."""
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self._profiles = json.load(f)
            print(f"Resuming — {len(self._profiles)} profiles already done.")
        return {int(k) for k in self._profiles}

    async def save(self, user_id: int, profile: str):
        """Atomically add a profile and flush to disk."""
        async with self._lock:
            self._profiles[str(user_id)] = profile
            self._flush()

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._profiles, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    @property
    def count(self) -> int:
        return len(self._profiles)


def load_data(
    train_path:  str = "data/processed/train_ratings.csv",
    movies_path: str = "data/raw/movies.dat",
) -> tuple[dict[int, dict], dict[int, str]]:
    train_df = pd.read_csv(train_path)
    train_df.columns = (
        train_df.columns.str.strip().str.lower()
        .str.replace(r"[^a-z0-9]", "_", regex=True)
    )
    if "userid"    in train_df.columns: train_df.rename(columns={"userid":    "user_id"},  inplace=True)
    if "movieid"   in train_df.columns: train_df.rename(columns={"movieid":   "movie_id"}, inplace=True)
    if "timestamp" in train_df.columns: train_df.drop(columns=["timestamp"],               inplace=True)

    movies_df = pd.read_csv(
        movies_path, sep="::", engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    movie_titles = dict(zip(movies_df["movie_id"], movies_df["title"]))

    user_ratings: dict[int, dict] = defaultdict(dict)
    for _, row in train_df.iterrows():
        user_ratings[int(row["user_id"])][int(row["movie_id"])] = float(row["rating"])

    return dict(user_ratings), movie_titles


def build_prompt(user_id: int, ratings: dict[int, float], movie_titles: dict[int, str]) -> str:
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    lines = [
        f"  - {movie_titles.get(mid, f'Movie {mid}')}: {r}/5"
        for mid, r in sorted_ratings
    ]
    return f"""You are a movie taste analyst. Based on the following movie ratings from user {user_id}, write a concise user profile (3-5 sentences) describing their movie preferences, favorite genres, and viewing patterns.

Ratings:
{chr(10).join(lines)}

Respond with only the profile text, no headers or extra formatting."""


async def generate_profile(
    sem: asyncio.Semaphore,
    checkpointer: ProfileCheckpointer,
    user_id: int,
    ratings: dict[int, float],
    movie_titles: dict[int, str],
):
    prompt = build_prompt(user_id, ratings, movie_titles)
    async with sem:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
        )
    profile_text = response.choices[0].message.content.strip()
    await checkpointer.save(user_id, profile_text)


async def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    checkpointer = ProfileCheckpointer(OUTPUT_PATH)
    done = checkpointer.load()

    print("Loading train data...")
    user_ratings, movie_titles = load_data()
    print(f"Found {len(user_ratings)} users in train set.")

    pending = {uid: r for uid, r in user_ratings.items() if uid not in done}
    print(f"Skipping {len(done)} done — {len(pending)} remaining.")

    if not pending:
        print("All profiles already generated.")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        generate_profile(sem, checkpointer, uid, ratings, movie_titles)
        for uid, ratings in pending.items()
    ]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating profiles"):
        await coro

    print(f"Done. {checkpointer.count} total profiles saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
