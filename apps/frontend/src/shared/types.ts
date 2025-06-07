export interface Brief {
  slug: string;
  date: {
    month: string;
    day: number;
    year: number;
  };
  id: number;
  title: string;
  content: string;
  createdAt: Date;
  totalArticles: number;
  totalSources: number;
  usedArticles: number;
  usedSources: number;
  model_author: string | null;
}
