import ProgressBar from 'progress';
import chalk from 'chalk';
import ora, { Ora } from 'ora';

export interface ProgressReporter {
  start(message: string): void;
  update(message: string): void;
  success(message: string): void;
  error(message: string): void;
  createProgressBar(total: number, message: string): ProgressBar;
}

export class ConsoleProgressReporter implements ProgressReporter {
  private spinner: Ora | null = null;

  start(message: string): void {
    this.spinner = ora(message).start();
  }

  update(message: string): void {
    if (this.spinner) {
      this.spinner.text = message;
    }
  }

  success(message: string): void {
    if (this.spinner) {
      this.spinner.succeed(message);
      this.spinner = null;
    } else {
      console.log(chalk.green(`✅ ${message}`));
    }
  }

  error(message: string): void {
    if (this.spinner) {
      this.spinner.fail(message);
      this.spinner = null;
    } else {
      console.error(chalk.red(`❌ ${message}`));
    }
  }

  createProgressBar(total: number, message: string): ProgressBar {
    return new ProgressBar(`${message} [:bar] :current/:total :percent :eta`, {
      complete: '█',
      incomplete: '░',
      width: 30,
      total
    });
  }
}

export class SilentProgressReporter implements ProgressReporter {
  start(_message: string): void {
    // Silent implementation for testing
  }

  update(_message: string): void {
    // Silent implementation for testing
  }

  success(_message: string): void {
    // Silent implementation for testing
  }

  error(_message: string): void {
    // Silent implementation for testing
  }

  createProgressBar(total: number, message: string): ProgressBar {
    return new ProgressBar(`${message} [:bar] :current/:total :percent :eta`, {
      complete: '█',
      incomplete: '░',
      width: 30,
      total
    });
  }
}

// Global instance for use across the application
export const progressReporter = new ConsoleProgressReporter(); 