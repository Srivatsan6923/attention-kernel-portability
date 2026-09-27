// @ts-check

import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import sitemap from '@astrojs/sitemap';
import { defineConfig } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// https://astro.build/config
export default defineConfig({
	site: 'https://srivatsan6923.github.io',
	// The site is served from a subfolder of the Jekyll portfolio. Jekyll skips
	// folders that start with '_', so the assets folder is renamed.
	base: '/projects/attention-kernel-portability',
	build: { assets: 'astro-assets' },
	integrations: [mdx(), sitemap()],
	markdown: {
		// remark-math parses $...$ and $$...$$, and rehype-katex renders them at
		// build time. The KaTeX stylesheet is imported in BaseHead.astro.
		processor: unified({
			remarkPlugins: [remarkMath],
			rehypePlugins: [rehypeKatex],
		}),
	},
});
