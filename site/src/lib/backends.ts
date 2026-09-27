// One colour per implementation family, shared by every chart. The values match
// the --backend-* tokens in src/styles/tokens.css. SVG charts need the literal
// hex, so it is repeated here.
//
// Every implementation id in public/data/web.json must be listed. colourOf
// throws on an unknown id so a new implementation cannot slip in uncoloured.

export const BACKEND_COLOR = {
	reference: '#9e9e9e',
	inductor: '#7e57c2',
	sdpa: '#1769aa',
	triton: '#2a9d8f',
	fa2: '#e08c3a',
	fa3: '#c4562a',
	flashinfer: '#2e9e5b',
} as const;

export const BACKEND_LABEL = {
	reference: 'PyTorch reference',
	inductor: 'TorchInductor',
	sdpa: 'SDPA',
	triton: 'Triton',
	fa2: 'FlashAttention-2',
	fa3: 'FlashAttention-3',
	flashinfer: 'FlashInfer',
} as const;

export type Backend = keyof typeof BACKEND_COLOR;

export const IMPL_BACKEND: Record<string, Backend> = {
	'P0-naive': 'reference',
	'D0-naive-kv': 'reference',
	'P1-inductor': 'inductor',
	'P1-inductor-nofuse': 'inductor',
	'P1-inductor-where': 'inductor',
	'D1-inductor': 'inductor',
	'P2b-sdpa-mem-eff': 'sdpa',
	'P2c-sdpa-flash': 'sdpa',
	'P2d-sdpa-cudnn': 'sdpa',
	'D2-sdpa': 'sdpa',
	'P3-triton': 'triton',
	'P4-fa2': 'fa2',
	'D3-fa-kvcache': 'fa2',
	'D6-fa-prefill-at-1': 'fa2',
	'P4h-fa3': 'fa3',
	'D4-flashinfer': 'flashinfer',
};

export function colourOf(impl: string): string {
	const backend = IMPL_BACKEND[impl];
	if (!backend) throw new Error(`No backend colour mapped for implementation "${impl}"`);
	return BACKEND_COLOR[backend];
}

// Readable name for each implementation id, used wherever an id is shown.
export const IMPL_NAME: Record<string, string> = {
	'P0-naive': 'PyTorch reference',
	'P1-inductor': 'torch.compile',
	'P1-inductor-nofuse': 'torch.compile, no pattern matching',
	'P1-inductor-where': 'torch.compile, torch.where mask',
	'P2b-sdpa-mem-eff': 'SDPA memory-efficient',
	'P2c-sdpa-flash': 'SDPA FlashAttention',
	'P2d-sdpa-cudnn': 'SDPA cuDNN',
	'P3-triton': 'Triton tutorial kernel',
	'P4-fa2': 'FlashAttention-2',
	'P4h-fa3': 'FlashAttention-3',
	'D0-naive-kv': 'PyTorch reference',
	'D1-inductor': 'torch.compile',
	'D2-sdpa': 'SDPA',
	'D3-fa-kvcache': 'FlashAttention-2 KV cache',
	'D4-flashinfer': 'FlashInfer',
	'D6-fa-prefill-at-1': 'FlashAttention-2, q=1',
};

export const nameOf = (impl: string): string => IMPL_NAME[impl] ?? impl;
