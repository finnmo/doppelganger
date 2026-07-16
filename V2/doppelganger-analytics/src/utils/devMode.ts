import chalk from 'chalk';

// Configuration
const DEV_MODE = process.env.NODE_ENV === 'development';
const DEV_MESSAGE_LIMIT = 2000;  // Total messages to import in dev mode

export function isDevMode(): boolean {
  return DEV_MODE;
}

export function getDevMessageLimit(): number {
  return DEV_MESSAGE_LIMIT;
}

// Apply dev mode limit only during import phase
export function applyDevModeLimit<T>(data: T[], entityName: string = 'items'): T[] {
  if (!DEV_MODE) {
    return data;
  }

  const limit = DEV_MESSAGE_LIMIT;
  
  if (data.length > limit) {
    console.log(chalk.yellow(`⚠️  Development mode: Importing only ${limit} ${entityName} (${data.length} total)`));
    return data.slice(0, limit);
  }

  console.log(chalk.blue(`🔧 Development mode: Importing ${data.length} ${entityName}`));
  return data;
}

export function logDevModeStatus(): void {
  if (DEV_MODE) {
    console.log(chalk.yellow(`🔧 Development mode enabled - importing limited to ${DEV_MESSAGE_LIMIT} messages total`));
  }
} 