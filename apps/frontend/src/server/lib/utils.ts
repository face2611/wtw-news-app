export const MONTH_NAMES = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
];

export interface FormattedDate {
  month: string;
  day: number;
  year: number;
}

export function formatReportDate(date: Date): FormattedDate {
  return {
    month: MONTH_NAMES[date.getUTCMonth()],
    day: date.getUTCDate(),
    year: date.getUTCFullYear(),
  };
}

export function generateReportSlug(date: Date): string {
  const { month, day, year } = formatReportDate(date);
  return `${month.toLowerCase()}-${day}-${year}`;
}

export function ensureDate(dateInput: Date | string | null | undefined): Date {
  return dateInput ? new Date(dateInput) : new Date();
}
