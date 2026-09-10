// Global site data.

export const SITE_TITLE = 'Attention Kernel Portability Across GPU Architectures';
export const SITE_DESCRIPTION =
	'A measured comparison of attention implementations across A100, A10, L40, L40S and H100, with the separation, dispatch and provenance caveats the data carries.';
export const SITE_AUTHOR = 'Srivatsan Sarvesan';

export const REPO_URL = 'https://github.com/Srivatsan6923/attention-kernel-portability';

// Header nav. Absolute paths resolve to the portfolio this page is mounted in.
export const NAV = [
	{ label: 'Home', href: '/' },
	{ label: 'Projects', href: '/projects/' },
	{ label: 'Publications', href: '/publications/' },
	{ label: 'CV', href: '/cv/' },
	{ label: 'GitHub', href: REPO_URL },
];
