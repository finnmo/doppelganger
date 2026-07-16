import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { assertLocalRequest } from '@/lib/server/localOnly';
import { closeAnalyticsDb, resolveAnalyticsDbPath } from '@/lib/server/analyticsDb';
import { getPipelineStatus, resolveDashDataDirs } from '@/lib/server/pipelineStatus';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const WIPE_CONFIRM = 'DELETE MY DATA';

export async function GET(req: NextRequest) {
  const blocked = assertLocalRequest(req);
  if (blocked) return blocked;

  try {
    const status = getPipelineStatus();
    const configDir = path.join(os.homedir(), '.doppelgaenger-analytics');
    const secretsPath = path.join(configDir, 'secrets.enc.json');
    return NextResponse.json({
      ...status,
      configDir,
      secretsPresent: fs.existsSync(secretsPath),
      dashDataDirs: resolveDashDataDirs(),
      wipeConfirmPhrase: WIPE_CONFIRM,
      notes: [
        'Message history and analytics stay on this machine by default.',
        'API keys are stored encrypted under ~/.doppelgaenger-analytics/ and are never included in exports.',
        'Export and wipe are localhost-only unless DOPPELGANGER_ALLOW_REMOTE=1.',
      ],
    });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Failed to load privacy status' },
      { status: 500 }
    );
  }
}

/** Download analytics JSON bundle (no secrets, no raw message DB by default). */
export async function POST(req: NextRequest) {
  const blocked = assertLocalRequest(req);
  if (blocked) return blocked;

  try {
    const body = (await req.json().catch(() => ({}))) as {
      action?: 'export' | 'wipe';
      confirm?: string;
      includeDatabase?: boolean;
    };

    const action = body.action ?? 'export';

    if (action === 'export') {
      const files: Record<string, unknown> = {};
      const dirs = resolveDashDataDirs();
      for (const dir of dirs) {
        for (const name of fs.readdirSync(dir)) {
          if (!name.endsWith('.json')) continue;
          // Prefer first occurrence (env / dash-data before public copies)
          if (files[name] !== undefined) continue;
          try {
            files[name] = JSON.parse(fs.readFileSync(path.join(dir, name), 'utf8'));
          } catch {
            // skip corrupt
          }
        }
      }

      const payload: Record<string, unknown> = {
        exportedAt: new Date().toISOString(),
        kind: 'doppelgaenger-analytics-export',
        version: 1,
        includesSecrets: false,
        includesMessageDatabase: false,
        pipeline: getPipelineStatus(),
        files,
      };

      if (body.includeDatabase) {
        const dbPath = resolveAnalyticsDbPath();
        if (fs.existsSync(dbPath)) {
          const buf = fs.readFileSync(dbPath);
          payload.includesMessageDatabase = true;
          payload.databaseBase64 = buf.toString('base64');
          payload.databaseFileName = path.basename(dbPath);
          payload.warning =
            'This export includes your full message database. Treat the file as sensitive.';
        }
      }

      const stamp = new Date().toISOString().slice(0, 10);
      return new NextResponse(JSON.stringify(payload), {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Content-Disposition': `attachment; filename="doppelgaenger-export-${stamp}.json"`,
          'Cache-Control': 'no-store',
        },
      });
    }

    if (action === 'wipe') {
      if (body.confirm !== WIPE_CONFIRM) {
        return NextResponse.json(
          {
            error: `Type exact phrase "${WIPE_CONFIRM}" to confirm wipe.`,
            wipeConfirmPhrase: WIPE_CONFIRM,
          },
          { status: 400 }
        );
      }

      closeAnalyticsDb();

      const dbPath = resolveAnalyticsDbPath();
      for (const suffix of ['', '-wal', '-shm']) {
        const p = `${dbPath}${suffix}`;
        if (fs.existsSync(p)) fs.unlinkSync(p);
      }

      for (const dir of resolveDashDataDirs()) {
        for (const name of fs.readdirSync(dir)) {
          if (name.endsWith('.json')) {
            fs.unlinkSync(path.join(dir, name));
          }
        }
      }

      return NextResponse.json({
        ok: true,
        wiped: ['message database', 'dash-data JSON', 'dashboard public/data JSON'],
        preserved: ['API keys in ~/.doppelgaenger-analytics/secrets.enc.json'],
        note: 'Clear API keys separately from API settings if desired.',
      });
    }

    return NextResponse.json({ error: 'Unknown action' }, { status: 400 });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : 'Privacy action failed' },
      { status: 500 }
    );
  }
}
