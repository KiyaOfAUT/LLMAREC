# fix_empty_movie_profiles.py

import asyncio
import json
import os
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.gapgpt.app/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "sk-TACuXW1f88MxKsya3KaFtzBFmPw3PewcEYFCTCvVeNMGuRjm")
LLM_MODEL    = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
CONCURRENCY  = int(os.getenv("CONCURRENCY", "10"))
OUTPUT_PATH  = "data/profiles/movies.json"
MOVIES_PATH  = "data/raw/movies.dat"

client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
_lock = asyncio.Lock()


def load_movies() -> dict[int, dict[str, str]]:
    df = pd.read_csv(
        MOVIES_PATH, sep="::", engine="python",
        names=["movie_id", "title", "genres"], encoding="latin-1",
    )
    return {
        int(row["movie_id"]): {"title": row["title"].strip(), "genres": row["genres"].strip()}
        for _, row in df.iterrows()
    }


def build_prompt(title: str, genres: str) -> str:
    genre_str = ", ".join(genres.split("|")) if genres else "N/A"
    return (
        f'You are a knowledgeable film expert. Based solely on your knowledge of cinema, '
        f'write a concise (3–5 sentences) profile for the movie titled "{title}", '
        f'which falls under the genres: {genre_str}. Include key themes, style, notable aspects, '
        f'and what kind of audience might enjoy it.\n\n'
        f'Respond with only the profile text—no titles, headers, or extra formatting.'
    )


async def fix_one(
    sem: asyncio.Semaphore,
    profiles: dict[str, str],
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

    async with _lock:
        profiles[str(movie_id)] = profile
        # atomic flush after each fix
        tmp = OUTPUT_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        os.replace(tmp, OUTPUT_PATH)


async def main():
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        profiles: dict[str, str] = json.load(f)

    empty_ids = [int(k) for k, v in profiles.items() if not v or not v.strip()]
    print(f"Found {len(empty_ids)} empty profiles: {empty_ids}")

    if not empty_ids:
        print("Nothing to fix.")
        return

    movies = load_movies()
    missing = [mid for mid in empty_ids if mid not in movies]
    if missing:
        print(f"[warn] These IDs not found in movies.dat: {missing}")

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        fix_one(sem, profiles, mid, movies[mid])
        for mid in empty_ids if mid in movies
    ]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fixing"):
        await coro

    print(f"Done. Patched {len(tasks)} profiles in {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
