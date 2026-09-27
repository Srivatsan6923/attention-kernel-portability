# Article site

The Astro source for the project article. Every number on the page is read
from `public/data/web.json`, which `python -m akp.webdata` writes.

| Command | Action |
| :-- | :-- |
| `npm install` | Install dependencies |
| `npm run dev` | Start a local dev server at `localhost:4321` |
| `npm run build` | Build the site into `dist/` |
| `npm run deploy` | Build, then copy `dist/` into `../../Portfolio/srivatsan6923.github.io/projects/attention-kernel-portability/` |

Use `npm run deploy` instead of editing the portfolio copy by hand, so its
numbers stay in sync with `web.json`.
