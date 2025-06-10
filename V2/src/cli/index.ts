#!/usr/bin/env node
import yargs from 'yargs';

yargs
  .command('import <zip>', 'Import Instagram zip', () => {}, argv => {
    console.log('Importing', argv.zip);
  })
  .demandCommand()
  .help()
  .argv;
