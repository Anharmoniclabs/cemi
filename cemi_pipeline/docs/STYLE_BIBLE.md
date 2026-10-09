# Cemí: visual style contract

**Canonical direction:** Original artist's hand drawings -> cleaner finished graphic animation -> restrained anime polish -> Option B 'Moody Urban Dusk'. No generic anime/pretty-boy homogenization.

## Non-negotiable geometry

- Each character must preserve their own face (jaw, nose, brows, eye spacing, skin tone, hairline, height and body mass). Facial construction uses the author's boxy/organic asymmetric shapes.
- Ink: distinct line-widths, crisp outer contour, lighter internal lines. The cartoon should be *complete*, not over-sketched.
- Hair: group curls/locs into coherent masses while keeping irregular artist-made silhouettes.
- Eyes: graphic lids and selective catchlights; never default to exaggerated glossy anime eyes.
- Keep shapes animation-readable from full-body distance; cel shadows grouped in two or three levels.
- **No gratuitous text**: no inspirational quotes, floating slogans, graffiti words, wall lettering, brand logos, caption cards, typography, or speech-bubble text inside scene generations. Production notes go in JSON, not in pictures. Only explicit story-required signs or props may have writing, preferably overlaid accurately in post.

## Option B grade

Night-blue #1B2634; muted steel #35465B, #6C8597; amber edge light #E7783D; brown #715043; warm charcoal #211C1C. Keep natural skin diversity accurate. For emotional scenes, alter light intensity rather than changing character pigment.

## Character Bible wins over concept drift

The earlier curly-haired slim Manny sheet and later curly-haired JC concepts are visual explorations, **not verified story canon**. The supplied book's locked looks govern identities. Manny is 25, 6'3", massively built, close-cropped hair (explicit adaptation choice), and plays conga. JC is 25, lean, short dark curls faded at sides, headphones and coin. If user approves a different adaptation, update `configs/characters.json` with that decision first.

## Dataset hygiene

1. **Only user-authored or explicitly approved refined artworks**. Uploaded sample photos, novel pages, AI generations, and earlier character sheets are not automatically approved.
2. Image-level manifest entries must identify origin and a non-empty content caption. Avoid placing bad lettering, watermarks, checkerboards or collage UI in training set: these artifacts otherwise become part of learned 'style'.
3. Crop each image to single, finished artwork; not a contact sheet/lettered mood board. Preserve original file offline.
4. Balance characters, gender, ages, ethnicity, settings, outfits and shot sizes so one recurring male face doesn't become the global style.
5. Reserve at least 10-20% approved samples for held-out evaluation; `prepare` currently copies *everything*, so curate an explicit training-only manifest.
6. Do not publish images, raw datasets or trained weights without the artist's approval. `data/` and `models/` are gitignored.

## 3 acceptance gates

**Style:** recognizable author geometry, clean finish, Option B grade, no free-floating words. **Identity:** identifiable character in front/side/three-quarter view, across expressions/costumes, no swapped appearances. **Animation:** no face/hair/clothing flicker over movement, stable props, anatomy and camera continuity. Human art-director approval required before promoting any output to canonical set.
