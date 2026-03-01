"""Prompt engineering strategies for enhanced lyrics generation."""

import json
from pathlib import Path

GENRE_SYSTEM_PROMPTS = {
    "pop": (
        "You are a hit-making pop songwriter. Your lyrics are catchy, emotionally resonant, "
        "and designed for massive sing-along appeal. You excel at crafting memorable hooks, "
        "relatable themes, and upbeat energy. Think of the best pop songwriters: Max Martin, "
        "Sia, and Jack Antonoff. Your melodies are implied through rhythmic, syllable-conscious writing."
    ),
    "country": (
        "You are a Nashville-caliber country songwriter with a gift for heartfelt storytelling, "
        "vivid imagery, and singable melodies. Your lyrics paint pictures of real life -- dusty roads, "
        "small towns, love and loss. You balance wit with sincerity, and every chorus feels like home. "
        "Think Kris Kristofferson's poetry, Dolly Parton's warmth, and Chris Stapleton's raw honesty."
    ),
    "rock": (
        "You are a rock songwriter channeling raw emotion and anthemic power. Your lyrics hit hard "
        "with visceral imagery, driving rhythm, and unapologetic intensity. You write for the stage -- "
        "every line should feel like it could be screamed by a crowd. Draw from the traditions of "
        "classic rock storytelling, punk directness, and alternative rock's emotional honesty."
    ),
    "indie": (
        "You are an indie songwriter with a gift for poetic observation and quiet emotional devastation. "
        "Your lyrics find beauty in mundane details, use unexpected metaphors, and create intimate "
        "atmospheres. You favor specificity over generality -- a particular street name over 'the road,' "
        "a specific time over 'someday.' Think Elliott Smith's vulnerability, Phoebe Bridgers' imagery, "
        "and Bon Iver's atmospheric language."
    ),
}

STRUCTURAL_INSTRUCTIONS = (
    "Structure the song with clear sections: verses, a chorus, and optionally a bridge or pre-chorus. "
    "Label each section (e.g., [Verse 1], [Chorus], [Bridge]). "
    "The chorus should be the emotional and melodic peak -- make it memorable and repeatable. "
    "Use consistent rhyme schemes within sections. Vary line lengths to create rhythmic interest. "
    "The song should feel complete with a beginning, build, climax, and resolution."
)

FEW_SHOT_EXAMPLES = {
    "pop": (
        "[Verse 1]\n"
        "Woke up to the sound of your name on my lips\n"
        "Golden light through the blinds, coffee drips\n"
        "Every morning's a movie when you're the star\n"
        "Don't need a map, you're my true north by far\n\n"
        "[Chorus]\n"
        "Light me up, light me up like neon\n"
        "Every heartbeat's a song we keep singing on\n"
        "Light me up, can't get enough\n"
        "You're the rush, you're the high, you're the love"
    ),
    "country": (
        "[Verse 1]\n"
        "Gravel road kicks dust up past the mailbox\n"
        "Screen door swinging where my grandma used to talk\n"
        "About the way the world was, how the river bends\n"
        "How the best things in this life don't ever end\n\n"
        "[Chorus]\n"
        "It's a little bit of dirt, a little bit of grace\n"
        "The kind of love that time can't erase\n"
        "Sunsets burning gold on a farmer's field\n"
        "This is what it means to be real"
    ),
    "rock": (
        "[Verse 1]\n"
        "Gasoline and summer heat\n"
        "Burning rubber, dead-end street\n"
        "Every scar's a story told\n"
        "In leather jackets, hearts of gold\n\n"
        "[Chorus]\n"
        "We are the thunder, we are the flame\n"
        "Screaming louder, they'll know our name\n"
        "Burn it down, we'll build it again\n"
        "We are the thunder, we never end"
    ),
    "indie": (
        "[Verse 1]\n"
        "Tuesday rain on Mulberry Street\n"
        "Your umbrella's broken, soaking wet feet\n"
        "We ducked into that bookshop on the corner\n"
        "Where the owner's cat sleeps on Faulkner\n\n"
        "[Chorus]\n"
        "And these are the hours that haunt me most\n"
        "Not the grand gestures, but the almost\n"
        "The way you laughed at nothing, the quiet after\n"
        "Every small disaster"
    ),
}


def get_prompts_config() -> dict:
    """Return current prompts as a serializable dict."""
    return {
        "system_prompts": dict(GENRE_SYSTEM_PROMPTS),
        "structural_instructions": STRUCTURAL_INSTRUCTIONS,
        "few_shot_examples": dict(FEW_SHOT_EXAMPLES),
    }


def load_prompts_config(path: str | Path) -> dict:
    """Load a prompts config from a JSON artifact file."""
    with open(path) as f:
        return json.load(f)


def save_prompts_config(config: dict, path: str | Path) -> None:
    """Save a prompts config to a JSON artifact file."""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
