#!/usr/bin/env node
import pkg from 'enquirer';
const { prompt } = pkg;
import chalk from 'chalk';
import ora from 'ora';
import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import { importArchive } from '../importer.js';
import { generateDashboards } from '../generator.js';
import { computePersonaEval } from '../processors/personaEval.js';
import { exportPersonaFineTune } from '../processors/personaFineTuneExport.js';
import { resolveDataPaths, describeDataExpectations } from '../utils/resolveDataPath.js';


async function runImport(zipPath: string, options?: { generate?: boolean }) {
  const spinner = ora(`Importing from ${zipPath}`).start();
  try {
    await importArchive(zipPath);
    spinner.succeed(chalk.green('Import complete!'));
  } catch (err) {
    spinner.fail(chalk.red('Import failed: ') + (err instanceof Error ? err.message : String(err)));
    throw err;
  }

  // Default: refresh analytics after every dump so persona profiles stay current.
  const shouldGenerate = options?.generate !== false;
  if (shouldGenerate) {
    console.log(chalk.blue('↻ Regenerating analytics after import…'));
    await runGenerate();
  } else {
    console.log(
      chalk.yellow(
        'Skipped generate (`--no-generate`). Persona profiles may be stale until you run `npm run generate-metrics`.'
      )
    );
  }
}

async function runGenerate() {
  const genSpinner = ora('Generating dashboards…').start();
  try {
    await generateDashboards();
    genSpinner.succeed(chalk.green('Dashboards ready!'));
  } catch (err) {
    genSpinner.fail(chalk.red('Generation failed: ') + (err instanceof Error ? err.message : String(err)));
    throw err;
  }
}

async function runDashboard() {
  console.log(chalk.blue('🚀 Starting analytics dashboard...'));
  
  try {
    // Import path and spawn modules
    const path = await import('path');
    const { spawn } = await import('child_process');
    
    // Check if dashboard directory exists
    const fs = await import('fs');
    const dashboardPath = path.resolve('dashboard');
    
    if (!fs.existsSync(dashboardPath)) {
      console.log(chalk.red('❌ Dashboard not found. Please ensure the dashboard is set up.'));
      return;
    }

    // Sync latest data to dashboard: dash-data is the single source of truth,
    // so stale files in public/data are removed rather than accumulating
    console.log(chalk.blue('📊 Copying latest data to dashboard...'));
    const dashDataPath = path.resolve('dash-data');
    const publicDataPath = path.resolve('dashboard/public/data');

    if (!fs.existsSync(publicDataPath)) {
      fs.mkdirSync(publicDataPath, { recursive: true });
    }

    if (fs.existsSync(dashDataPath)) {
      const files = fs.readdirSync(dashDataPath).filter(file => file.endsWith('.json'));
      const fileSet = new Set(files);

      for (const staleFile of fs.readdirSync(publicDataPath)) {
        if (staleFile.endsWith('.json') && !fileSet.has(staleFile)) {
          fs.rmSync(path.join(publicDataPath, staleFile));
        }
      }
      for (const file of files) {
        fs.copyFileSync(path.join(dashDataPath, file), path.join(publicDataPath, file));
      }
      console.log(chalk.green(`✅ Synced ${files.length} data files to dashboard.`));
    } else {
      console.log(chalk.yellow('⚠️  No dashboard data found. Run "generate" first to create analytics data.'));
    }

    // Pick a free port ourselves so we can print the real URL. Probe on
    // 0.0.0.0 (not 127.0.0.1): something may already be bound on :::3000 while
    // IPv4 loopback still looks free — which previously made us claim :3000
    // and then Next failed with EADDRINUSE.
    const { createServer } = await import('net');
    const port = await new Promise<number>((resolve, reject) => {
      const tryPort = (candidate: number) => {
        if (candidate > 3010) {
          reject(new Error('No free port found between 3000 and 3010'));
          return;
        }
        const server = createServer();
        server.unref();
        server.on('error', () => tryPort(candidate + 1));
        server.listen(candidate, '0.0.0.0', () => {
          const chosen = (server.address() as { port: number }).port;
          server.close(() => resolve(chosen));
        });
      };
      tryPort(3000);
    });

    // Start the Next.js development server
    console.log(chalk.blue('🌐 Starting dashboard server...'));
    const dashboardProcess = spawn('npm', ['run', 'dev', '--', '-p', String(port)], {
      cwd: dashboardPath,
      stdio: 'inherit',
      // npm is npm.cmd on Windows and can only be spawned through a shell
      shell: process.platform === 'win32',
      env: { ...process.env, PORT: String(port), HOSTNAME: '127.0.0.1' }
    });

    console.log(chalk.green('🎉 Dashboard is starting! It will be available at:'));
    console.log(chalk.cyan(`   → http://localhost:${port}`));
    console.log('');
    console.log(chalk.gray('Press Ctrl+C to stop the dashboard server.'));

    // Handle process termination
    process.on('SIGINT', () => {
      console.log(chalk.yellow('\n⏹️  Stopping dashboard server...'));
      dashboardProcess.kill('SIGINT');
      process.exit(0);
    });

    // Wait for the process to end — non-zero (or abrupt crash) must surface
    // as a failure so callers don't think the dashboard is running.
    const exitCode = await new Promise<number>((resolve) => {
      dashboardProcess.on('close', (code) => resolve(code ?? 1));
      dashboardProcess.on('error', () => resolve(1));
    });
    if (exitCode !== 0) {
      throw new Error(`Dashboard process exited with code ${exitCode}`);
    }

  } catch (err) {
    console.error(chalk.red('❌ Failed to start dashboard: ') + (err instanceof Error ? err.message : String(err)));
    process.exitCode = 1;
  }
}

async function interactiveMenu() {
  while (true) {
    const { command } = await prompt<{ command: string }>({
      type: 'select',
      name: 'command',
      message: 'What would you like to do?',
      choices: [
        { name: 'start',     message: '🚀 Start (auto-detect → import → generate → dashboard)' },
        { name: 'import',    message: '📥 Import messaging archive' },
        { name: 'generate',  message: '📊 Generate analytics data (fast mode)' },
        { name: 'dashboard', message: '🌐 Start analytics dashboard' },
        { name: 'exit',      message: '🚪 Exit' },
      ]
    });

    if (command === 'exit') {
      console.log(chalk.green('Goodbye! 👋'));
      process.exit(0);
    }

    try {
      if (command === 'start') {
        await runStart(undefined);
      } else if (command === 'import') {
        const { zip } = await prompt<{ zip: string }>({
          type: 'input',
          name: 'zip',
          message: 'Path to your export (ZIP, folder, _chat.txt, or chat.db):',
        });
        await runImport(zip);
      } else if (command === 'generate') {
        await runGenerate();
      } else if (command === 'dashboard') {
        await runDashboard();
      }
    } catch {
      // Error already reported by the runner; return to the menu.
    }

    // After any command completes, loop back to menu
    console.log();
  }
}

async function runStart(explicitData: string | string[] | undefined) {
  const dataPaths = resolveDataPaths(explicitData);
  if (dataPaths.length === 0) {
    console.log(chalk.yellow('No export found in the project root.'));
    console.log(chalk.gray(describeDataExpectations()));
    throw new Error('No data export found. Pass --data=/path/to/export or place a file in the project root.');
  }
  for (const dataPath of dataPaths) {
    console.log(chalk.blue(`📂 Using export: ${dataPath}`));
    // Import only — generate once after all dumps land.
    await runImport(dataPath, { generate: false });
  }
  if (dataPaths.length > 1) {
    console.log(chalk.green(`✅ Imported ${dataPaths.length} exports (platforms merged in one database)`));
  }
  await runGenerate();
  await runDashboard();
}

async function main() {
  // First see if a yargs subcommand was passed. Handlers must return their
  // promise so yargs waits for completion (otherwise the process can exit
  // before the async work finishes).
  const argv = await yargs(hideBin(process.argv))
    .scriptName('doppel-analytics')
    .option('data', {
      type: 'string',
      describe: 'Path to export (ZIP, folder, _chat.txt, or chat.db)',
    })
    .command('start', 'Auto-detect export → import → generate → dashboard', () => {}, async args => {
      await runStart(args.data as string | undefined);
    })
    .command(
      'import <zip>',
      'Import messaging archive (regenerates analytics by default)',
      (y) =>
        y.option('generate', {
          type: 'boolean',
          default: true,
          describe: 'Run generate-metrics after a successful import (use --no-generate to skip)',
        }),
      async (args) => {
        await runImport(args.zip as string, { generate: args.generate !== false });
      }
    )
    .command('generate', 'Generate analytics data (fast mode)', () => {}, async () => {
      await runGenerate();
    })
    .command(
      'persona-eval',
      'Build held-out persona eval set (add --live to score with Claude)',
      (y) =>
        y
          .option('live', {
            type: 'boolean',
            default: false,
            describe: 'Generate replies with Claude and score vs held-out actuals',
          })
          .option('sender', {
            type: 'string',
            describe: 'Only evaluate this sender',
          })
          .option('limit', {
            type: 'number',
            default: 25,
            describe: 'Held-out pairs per sender',
          }),
      async (args) => {
        await computePersonaEval({
          live: Boolean(args.live),
          pairsPerSender: Number(args.limit) || 25,
          senderFilter: typeof args.sender === 'string' ? args.sender : undefined,
          maxSenders: 12,
        });
      }
    )
    .command(
      'persona-finetune-export',
      'Export chat JSONL for Tier-4 fine-tuning (does not train)',
      (y) =>
        y
          .option('limit', {
            type: 'number',
            default: 200,
            describe: 'Max pairs per sender',
          })
          .option('senders', {
            type: 'number',
            default: 20,
            describe: 'Max senders to include',
          }),
      async (args) => {
        await exportPersonaFineTune({
          maxPairsPerSender: Number(args.limit) || 200,
          maxSenders: Number(args.senders) || 20,
          preferWithYou: true,
        });
      }
    )
    .command('dashboard', 'Start analytics dashboard', () => {}, async () => {
      await runDashboard();
    })
    .help()
    .argv;

  // If no command was passed, run interactive mode
  if (argv._.length === 0) {
    console.log(chalk.cyan('🎯 Doppelgänger Analytics CLI'));
    console.log(chalk.gray('WhatsApp, Messenger, iMessage, Instagram — import, analyze, dashboard'));
    console.log();
    await interactiveMenu();
  }
}

main().catch(err => {
  console.error(chalk.red('Error:'), err);
  process.exit(1);
});
