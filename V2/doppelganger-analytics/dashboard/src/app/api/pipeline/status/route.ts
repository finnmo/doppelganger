import { NextResponse } from 'next/server';
import { getPipelineStatus } from '@/lib/server/pipelineStatus';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(getPipelineStatus());
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Failed to read pipeline status' },
      { status: 500 }
    );
  }
}
