import { localDateString } from '../../src/utils/dates.js';

describe('localDateString', () => {
  test('formats a timestamp as local YYYY-MM-DD', () => {
    // Construct a local-time date and confirm the same calendar day is returned
    const d = new Date(2024, 2, 15, 23, 30); // 15 Mar 2024, 11:30pm local
    expect(localDateString(d.getTime())).toBe('2024-03-15');
  });

  test('zero-pads month and day', () => {
    const d = new Date(2023, 0, 5, 9, 0); // 5 Jan 2023
    expect(localDateString(d.getTime())).toBe('2023-01-05');
  });

  test('uses local time, not UTC (late-night dates do not roll over)', () => {
    // 1 Jan 2024 00:30 local — under UTC this could read as 31 Dec depending on
    // offset; localDateString must report the local calendar day.
    const d = new Date(2024, 0, 1, 0, 30);
    expect(localDateString(d.getTime())).toBe('2024-01-01');
  });
});
