/**
 * Guard destructive / data-exfiltration routes to loopback only.
 * Prevents accidental exposure if the dashboard is ever bound beyond localhost.
 */

import { NextRequest, NextResponse } from 'next/server';

function isLoopbackHost(host: string | null): boolean {
  if (!host) return false;
  const hostname = host.split(':')[0]?.toLowerCase() ?? '';
  return (
    hostname === 'localhost' ||
    hostname === '127.0.0.1' ||
    hostname === '[::1]' ||
    hostname === '::1'
  );
}

export function assertLocalRequest(req: NextRequest): NextResponse | null {
  // Allow explicit override for trusted private deploys
  if (process.env.DOPPELGANGER_ALLOW_REMOTE === '1') {
    return null;
  }

  const host = req.headers.get('x-forwarded-host') ?? req.headers.get('host');
  const forwarded = req.headers.get('x-forwarded-for');
  const realIp = req.headers.get('x-real-ip');

  if (!isLoopbackHost(host)) {
    return NextResponse.json(
      {
        error:
          'This action is only available on localhost. Set DOPPELGANGER_ALLOW_REMOTE=1 only if you fully trust the network.',
      },
      { status: 403 }
    );
  }

  // If a proxy claimed a non-local client, refuse (unless allow-remote)
  const clientHint = (forwarded?.split(',')[0]?.trim() || realIp || '').toLowerCase();
  if (
    clientHint &&
    clientHint !== '127.0.0.1' &&
    clientHint !== '::1' &&
    clientHint !== 'localhost' &&
    !clientHint.startsWith('127.')
  ) {
    return NextResponse.json(
      {
        error:
          'Refusing remote client for privacy-sensitive action. Set DOPPELGANGER_ALLOW_REMOTE=1 only if intentional.',
      },
      { status: 403 }
    );
  }

  return null;
}
