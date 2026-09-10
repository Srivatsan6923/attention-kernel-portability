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
	// Served as a static subfolder of the Jekyll portfolio, so bundled asset URLs
	// need the mount point. Jekyll skips directories starting with '_', hence the
	// renamed assets dir.
	base: '/projects/attention-kernel-portability',
	build: { assets: 'astro-assets' },
	integrations: [mdx(), sitemap()],
	markdown: {
		// remark-math parses $...$ and $$...$$; rehype-katex renders them at build
		// time. KaTeX's stylesheet is imported locally in BaseHead.astro.
		processor: unified({
			remarkPlugins: [remarkMath],
			rehypePlugins: [rehypeKatex],
		}),
	},
});
